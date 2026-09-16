# -*- coding: utf-8 -*-
"""2048 训练器图形界面（8 进程并行版）

双击「训练AI.bat」或运行  python train_gui.py

- 8 个训练进程通过系统共享内存挂载同一份权重表（Hogwild 无锁更新），
  吞吐约等于单进程的 8 倍；内存仍只占一份（约 268MB）
- 模型统一由主进程落盘：暂停/停止/关闭窗口时保存，训练中每 60 秒自动快照
- 实时统计：进度数据每 0.4 秒聚合上报（局数、速度、会话最佳分/最大块），
  完整窗口统计（平均分、达成率、曲线）每 worker 500 局刷新一次
- 性能占用控制:
    优先级  全速(NORMAL) / 平衡(BELOW_NORMAL, 默认) / 空闲(IDLE)
    限速    全局 局/秒 上限滑块（内部均摊给各 worker），0 表示不限
- 训练进行时自动阻止系统休眠（屏幕仍可关闭省电），暂停/停止/关窗后恢复
"""

import json
import threading
import time
import tkinter as tk
from ctypes import POINTER, c_double, c_int, c_int64
from pathlib import Path

import ctypes

import numpy as np

import ai2048
from train2048 import HISTORY_PATH, MODEL_PATH, load_trained_games

BG = "#faf8ef"
CARD = "#bbada0"
CARD_BG2 = "#cdc1b4"
DARK = "#776e65"
LIGHT = "#f9f6f2"
ACCENT = "#8f7a66"
RED = "#f65e3b"
GOLD = "#d4a017"
FONT = "Microsoft YaHei UI"

N_WORKERS = 8
REPORT_EVERY = 250          # 每 worker 每 250 局上报一次完整窗口统计
                            # （8 进程下约 25~35 秒一条，兼顾及时性与样本量）
PROGRESS_INTERVAL = 0.4     # 进度上报间隔（秒）
SNAPSHOT_INTERVAL = 60_000  # 训练中自动快照间隔（毫秒）
SHM_NAME = "2048_td_weights"
SETTINGS_PATH = Path(__file__).resolve().parent / "settings.json"


def load_settings():
    try:
        return json.loads(SETTINGS_PATH.read_text("utf-8"))
    except Exception:
        return {}


def save_settings(d):
    try:
        SETTINGS_PATH.write_text(json.dumps(d, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
    except Exception:
        pass

# Windows 进程优先级
PRIORITY = {"normal": 0x20, "below": 0x4000, "idle": 0x40}

# SetThreadExecutionState 标志：训练期间阻止系统睡眠（屏幕允许关闭）
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def set_priority(name):
    ctypes.windll.kernel32.SetPriorityClass(
        ctypes.windll.kernel32.GetCurrentProcess(), PRIORITY[name])


def set_keep_awake(on):
    """训练进行时阻止 Windows 休眠/睡眠；停止/暂停后恢复系统默认。"""
    flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0)
    return ctypes.windll.kernel32.SetThreadExecutionState(flags)


class TrainGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("2048 训练器")
        self.configure(bg=BG)
        self.resizable(False, False)

        self.training = False
        self.paused = False
        self.w_view = None
        self._stats_buf = np.zeros((64, 10), dtype=np.int64)
        self._seg_buf = np.zeros(9, dtype=np.int64)
        self._last_total = 0
        self._last_gps_t = time.time()
        self.points = []          # 存档点 [(games, avg_score, rate_2048), ...]
        self.live_points = []     # 本会话实时采样点（仅内存）
        self._last_live_ts = 0.0
        self.last_stats = {}
        self.last_games = 0
        self._saving = False
        self.settings = load_settings()   # 持久化的 α/优先级/限速
        self._load_history()
        self._start_games = self.last_games   # 本次训练起点

        self._build_ui()
        self._fill_from_history()
        self._poll()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------- 历史 ----------
    def _fill_from_history(self):
        s = self.last_stats
        if not s:
            return
        self.card_vars["累计局数"].set(f"{s['games']:,}")
        self.card_vars["平均分(实时)"].set(f"{s['avg_score']:,.0f}")
        self.card_vars["最佳分"].set(f"{s['best_score']:,}")
        self.card_vars["最大块"].set(f"{s['max_tile']:,}")
        self.card_vars["≥8192 达成率"].set(f"{s.get('rate_8192', 0):.1%}")
        self.card_vars["≥16384 达成率"].set(f"{s.get('rate_16384', 0):.1%}")

    def _load_history(self):
        if HISTORY_PATH.exists():
            for line in HISTORY_PATH.read_text("utf-8").splitlines():
                try:
                    rec = json.loads(line)
                    if rec["games"] > self.last_games:
                        self.last_games = rec["games"]   # 局数对账用全部记录
                    if rec.get("count", 0) < 100:
                        continue   # 忽略中断残余的小窗口，其达成率无统计意义
                    self.points.append((rec["games"], rec["avg_score"],
                                         rec.get("rate_8192", 0), rec.get("rate_16384", 0)))
                    self.last_stats = rec
                except (ValueError, KeyError):
                    pass

    # ---------- 界面 ----------
    def _build_ui(self):
        outer = tk.Frame(self, bg=BG, padx=20, pady=14)
        outer.pack()

        head = tk.Frame(outer, bg=BG)
        head.pack(fill="x")
        tk.Label(head, text="2048 训练器", font=(FONT, 24, "bold"),
                 bg=BG, fg=DARK).pack(side="left")
        tk.Label(head, text=f"C++ {N_WORKERS} 线程", font=(FONT, 12),
                 bg=BG, fg=ACCENT).pack(side="left", padx=(10, 0), pady=14)

        cards = tk.Frame(outer, bg=BG)
        cards.pack(fill="x", pady=(10, 6))
        self.card_vars = {}
        for i, key in enumerate(("累计局数", "本次训练", "速度 局/秒", "平均分(实时)",
                                 "最佳分", "最大块", "≥8192 达成率", "≥16384 达成率")):
            self.card_vars[key] = tk.StringVar(value="—")
            self._card(cards, key, i)

        self.chart = tk.Canvas(outer, width=720, height=250, bg="white",
                               highlightthickness=1, highlightbackground=CARD_BG2)
        self.chart.pack(pady=6)
        self.draw_chart()

        ctrl = tk.Frame(outer, bg=BG)
        ctrl.pack(fill="x", pady=(4, 0))

        row1 = tk.Frame(ctrl, bg=BG)
        row1.pack(fill="x", pady=2)
        self.btn_main = tk.Button(row1, text="开始训练", command=self._on_main,
                                  bg=ACCENT, fg=LIGHT, activebackground="#9f8b77",
                                  activeforeground=LIGHT, relief="flat",
                                  font=(FONT, 13, "bold"), padx=22, pady=5,
                                  cursor="hand2", takefocus=0)
        self.btn_main.pack(side="left")
        self.btn_stop = tk.Button(row1, text="停止并保存", command=self.stop_training,
                                  bg=CARD, fg=LIGHT, activebackground=CARD_BG2,
                                  activeforeground=LIGHT, relief="flat",
                                  font=(FONT, 13, "bold"), padx=14, pady=5,
                                  cursor="hand2", takefocus=0, state="disabled")
        self.btn_stop.pack(side="left", padx=(8, 0))

        tk.Label(row1, text="    CPU 优先级:", font=(FONT, 11),
                 bg=BG, fg=DARK).pack(side="left")
        self.prio_var = tk.StringVar(value=self.settings.get("priority", "below"))
        for text, val in (("全速", "normal"), ("平衡", "below"), ("空闲", "idle")):
            tk.Radiobutton(row1, text=text, variable=self.prio_var, value=val,
                           command=self._send_priority, font=(FONT, 11),
                           bg=BG, fg=DARK, selectcolor="#eee4da",
                           activebackground=BG, activeforeground=DARK,
                           takefocus=0).pack(side="left", padx=(6, 0))

        row2 = tk.Frame(ctrl, bg=BG)
        row2.pack(fill="x", pady=(8, 2))
        tk.Label(row2, text="局/秒上限", font=(FONT, 11), bg=BG, fg=DARK).pack(side="left")
        self.speed_var = tk.IntVar(value=int(self.settings.get("speed", 0)))
        tk.Scale(row2, variable=self.speed_var, from_=0, to=30, orient="horizontal",
                 length=200, showvalue=False, command=self._send_speed,
                 bg=BG, fg=DARK, troughcolor=CARD_BG2, highlightthickness=0,
                 takefocus=0).pack(side="left", padx=8)
        self.speed_lbl = tk.Label(row2, text="不限速", font=(FONT, 11, "bold"),
                                  bg=BG, fg=ACCENT, width=6, anchor="w")
        self.speed_lbl.pack(side="left")
        tk.Label(row2, text="（0 = 不限速；全局上限，均摊给各进程）",
                 font=(FONT, 10), bg=BG, fg=DARK).pack(side="left", padx=(4, 0))

        tk.Label(row2, text="学习率 α", font=(FONT, 11), bg=BG, fg=DARK).pack(side="right")
        self.alpha_var = tk.StringVar(value=str(self.settings.get("alpha", 0.005)))
        self._alpha_box = tk.Spinbox(row2, from_=0.0005, to=0.02, increment=0.0005, width=7,
                   textvariable=self.alpha_var, command=self._send_alpha,
                   font=(FONT, 11), buttonbackground=CARD_BG2, relief="flat",
                   takefocus=0)
        self._alpha_box.bind("<Return>", lambda e: self._send_alpha())
        self._alpha_box.bind("<FocusOut>", lambda e: self._send_alpha())
        self._alpha_box.pack(side="right", padx=8)

        self.status = tk.Label(outer, text="就绪 | 模型: " + str(MODEL_PATH),
                               font=(FONT, 10), bg=BG, fg=DARK, anchor="w")
        self.status.pack(fill="x", pady=(8, 0))

    def _card(self, parent, title, index):
        box = tk.Frame(parent, bg=CARD, padx=14, pady=6)
        box.grid(row=index // 4, column=index % 4, padx=(0, 8), pady=4, sticky="n")
        tk.Label(box, text=title, font=(FONT, 10), bg=CARD, fg="#eee4da").pack()
        tk.Label(box, textvariable=self.card_vars[title], font=(FONT, 15, "bold"),
                 bg=CARD, fg=LIGHT, width=10).pack()

    # ---------- 曲线 ----------
    @staticmethod
    def _ema(values, alpha):
        """指数移动平均：压平采样噪声、保留趋势"""
        out, v = [], values[0]
        for x in values:
            v = alpha * x + (1 - alpha) * v
            out.append(v)
        return out

    def _merged_points(self):
        """历史存档点 + 本会话实时采样点，按局数排序"""
        if self.live_points:
            return sorted(self.points + self.live_points)
        return self.points

    def _append_live(self, games, avg, rate8192, rate16384):
        """训练中每 2 秒把实时滚动值采进曲线（仅内存，不写历史文件）"""
        now = time.time()
        if now - self._last_live_ts < 2.0:
            return
        self._last_live_ts = now
        self.live_points.append((games, avg, rate8192, rate16384))
        self.draw_chart()

    def draw_chart(self):
        cv = self.chart
        cv.delete("all")
        w, h = 720, 250
        pl, pr, pt, pb = 52, 52, 34, 26
        iw, ih = w - pl - pr, h - pt - pb
        cv.create_text(pl, 12, text="训练曲线（实线 = EMA 降噪 · 淡线 = 原始）",
                       font=(FONT, 10), fill=DARK, anchor="w")

        pts = self._merged_points()
        if len(pts) < 2:
            cv.create_text(w / 2, h / 2, text="暂无足够数据，开始训练后自动绘制",
                           font=(FONT, 12), fill=CARD)
            return

        games = [p[0] for p in pts]
        avgs = [p[1] for p in pts]
        rates = [p[2] for p in pts]
        rates16 = [p[3] for p in pts]
        max_avg = max(avgs) * 1.05 or 1.0
        # 平滑强度自适应：点越多压得越狠（点少时避免过度平滑抹掉趋势）
        alpha = max(0.05, 5.0 / len(avgs))
        sm_avg = self._ema(avgs, alpha)
        sm_rate = self._ema(rates, alpha)
        sm_rate16 = self._ema(rates16, alpha)

        def X(g):
            return pl + (g - games[0]) / max(games[-1] - games[0], 1) * iw

        def Y1(v):
            return pt + ih - v / max_avg * ih

        def Y2(v):
            return pt + ih - v * ih

        cv.create_line(pl, pt, pl, pt + ih, fill=CARD_BG2)
        cv.create_line(pl, pt + ih, pl + iw, pt + ih, fill=CARD_BG2)
        for frac in (0, 0.5, 1.0):
            y = pt + ih * (1 - frac)
            cv.create_line(pl, y, pl + iw, y, fill="#eee4da")
            cv.create_text(pl - 6, y, text=f"{int(max_avg * frac):,}",
                           font=(FONT, 9), fill=DARK, anchor="e")
            cv.create_text(pl + iw + 6, y, text=f"{frac:.0%}",
                           font=(FONT, 9), fill=DARK, anchor="w")
        cv.create_text(pl, pt + ih + 14, text=f"{games[0]:,}",
                       font=(FONT, 9), fill=DARK)
        cv.create_text(pl + iw, pt + ih + 14, text=f"{games[-1]:,}",
                       font=(FONT, 9), fill=DARK, anchor="e")
        cv.create_text(452, 12, text="— 平均分", font=(FONT, 9, "bold"),
                       fill=RED, anchor="w")
        cv.create_text(550, 12, text="8192率 —", font=(FONT, 9, "bold"),
                       fill=GOLD, anchor="w")
        cv.create_text(636, 12, text="16384率 —", font=(FONT, 9, "bold"),
                       fill="#3c8c8c", anchor="w")

        # 原始数据：淡细线；平滑主线：实粗线
        cv.create_line([c for g, a, _, _ in pts for c in (X(g), Y1(a))],
                       fill="#f0bfb0", width=1)
        cv.create_line([c for g, _, r, _ in pts for c in (X(g), Y2(r))],
                       fill="#ecdfb4", width=1)
        cv.create_line([c for g, _, _, r16 in pts for c in (X(g), Y2(r16))],
                       fill="#c3d9d9", width=1)
        cv.create_line([c for g, a in zip(games, sm_avg) for c in (X(g), Y1(a))],
                       fill=RED, width=2)
        cv.create_line([c for g, r in zip(games, sm_rate) for c in (X(g), Y2(r))],
                       fill=GOLD, width=2)
        cv.create_line([c for g, r16 in zip(games, sm_rate16) for c in (X(g), Y2(r16))],
                       fill="#3c8c8c", width=2)

    # ---------- 训练生命周期 ----------
    def _on_main(self):
        if not self.training:
            self.start_training()
        elif self.paused:
            self.paused = False
            ai2048.CPP.ai_train_set_pause(0)
            set_keep_awake(True)
            self.btn_main.config(text="暂停训练")
            self.set_status("训练中")
        else:
            self.paused = True
            ai2048.CPP.ai_train_set_pause(1)
            set_keep_awake(False)
            self.btn_main.config(text="继续训练")
            self._save_model_async("暂停")

    def start_training(self):
        if self.training:
            return
        if ai2048.CPP is None:
            self.set_status("未找到 engine.dll，请先运行  python build_engine.py")
            return
        self._start_games = self.last_games
        self.live_points.clear()
        self._last_total = 0
        self._last_gps_t = time.time()
        if MODEL_PATH.exists():
            self.w_view = ai2048.NTupleNet.load(MODEL_PATH).w   # 含旧表半迁移
        else:
            self.w_view = np.zeros(ai2048.N_TABLES * ai2048.TABLE, np.float32)

        n = ai2048.CPP.ai_train_start(
            ai2048.CPP._pf(self.w_view), N_WORKERS, float(self.alpha_var.get()))
        if n <= 0:
            self.set_status("启动 C++ 训练线程失败")
            return
        self.training = True

        self._send_priority()
        self._send_speed(str(self.speed_var.get()))
        self._send_alpha()
        set_keep_awake(True)
        self.btn_main.config(text="暂停训练")
        self.btn_stop.config(state="normal")
        self.set_status(f"训练中… {n} 线程 C++ · 已阻止系统休眠 · "
                        f"关闭窗口自动停止并保存")
        self.after(SNAPSHOT_INTERVAL, self._snapshot_loop)

    def stop_training(self):
        if not self.training:
            return

        def do_stop():
            ai2048.CPP.ai_train_stop()
            self.after(0, self._finish_stop)

        threading.Thread(target=do_stop, daemon=True).start()

    def _save_model_async(self, when):
        """把共享表快照到磁盘（后台线程，避免卡界面）。"""
        if self.w_view is None or self._saving:
            return
        self._saving = True
        snapshot = self.w_view.copy()

        def write():
            try:
                np.savez(MODEL_PATH, w=snapshot)
                try:
                    self.after(0, lambda: self.set_status(f"模型已保存（{when}）"))
                except RuntimeError:
                    pass   # 窗口已销毁
            except Exception as e:
                try:
                    self.after(0, lambda e=e: self.set_status(f"保存失败: {e}"))
                except RuntimeError:
                    pass
            finally:
                self._saving = False

        threading.Thread(target=write, daemon=True).start()

    def _snapshot_loop(self):
        if self.training:
            self._save_model_async("自动快照")
            self.after(SNAPSHOT_INTERVAL, self._snapshot_loop)

    def _save_settings(self):
        try:
            alpha = float(self.alpha_var.get())
        except ValueError:
            alpha = float(self.settings.get("alpha", 0.005))
        save_settings({"alpha": alpha,
                       "priority": self.prio_var.get(),
                       "speed": int(self.speed_var.get())})

    def _send_priority(self):
        set_priority(self.prio_var.get())
        self._save_settings()

    def _send_speed(self, _=None):
        v = self.speed_var.get()
        self.speed_lbl.config(text="不限速" if v == 0 else f"{v} 局/秒")
        if self.training:
            per = 0 if v == 0 else max(1, round(v / N_WORKERS))
            ai2048.CPP.ai_train_set_speed(float(per))
        self._save_settings()

    def _send_alpha(self):
        if self.training:
            try:
                ai2048.CPP.ai_train_set_alpha(float(self.alpha_var.get()))
            except ValueError:
                pass
        self._save_settings()

    def set_status(self, text):
        self.status.config(text=text + " | 模型: " + str(MODEL_PATH))

    # ---------- 统计轮询与聚合（C++ 训练线程） ----------
    def _poll(self):
        if self.training:
            self._aggregate()
        self.after(50, self._poll)

    def _aggregate(self):
        CPP = ai2048.CPP
        buf = self._stats_buf
        CPP.ai_train_stats(buf.ctypes.data_as(POINTER(c_int64)), 64)
        a = buf[:N_WORKERS]
        total = int(a[:, 0].sum())
        games = self._start_games + total
        now = time.time()
        if now - self._last_gps_t >= 1.0:      # 速度按 1 秒差分
            self._gps_val = (total - self._last_total) / (now - self._last_gps_t)
            self._last_total, self._last_gps_t = total, now
        self.card_vars["累计局数"].set(f"{games:,}")
        self.card_vars["本次训练"].set(f"{games - self._start_games:,}")
        self.card_vars["速度 局/秒"].set(f"{getattr(self, '_gps_val', 0):.0f}")
        self.card_vars["最佳分"].set(f"{a[:, 1].max():,}")
        self.card_vars["最大块"].set(f"{a[:, 2].max():,}")
        tn = int(a[:, 3].sum())                # 滚动窗口聚合 → 实时卡片
        if tn > 0:
            tsum = int(a[:, 4].sum())
            r8 = int(a[:, 7].sum())
            r16 = int(a[:, 9].sum())
            self.card_vars["平均分(实时)"].set(f"{tsum / tn:,.0f}")
            self.card_vars["≥8192 达成率"].set(f"{r8 / tn:.1%}")
            self.card_vars["≥16384 达成率"].set(f"{r16 / tn:.1%}")
            self._append_live(games, tsum / tn, r8 / tn, r16 / tn)
        # 段统计（每 worker 满 REPORT_EVERY 局）→ 历史与曲线
        for i in range(N_WORKERS):
            if a[i, 8] >= REPORT_EVERY:
                n = int(CPP.ai_train_seg_take(
                    i, self._seg_buf.ctypes.data_as(POINTER(c_int64))))
                if n >= 100:
                    g = self._seg_buf
                    rec = {"games": games, "count": n,
                           "avg_score": g[1] / n, "best_score": int(g[2]),
                           "max_tile": int(g[3]),
                           "rate_1024": g[4] / n, "rate_2048": g[5] / n,
                           "rate_4096": g[6] / n, "rate_8192": g[7] / n,
                           "rate_16384": g[8] / n}
                    with open(HISTORY_PATH, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + chr(10))
                    self.points.append((games, rec["avg_score"],
                                        rec["rate_8192"], rec["rate_16384"]))
                    self.draw_chart()
                self.last_games = games

    def _finish_stop(self):
        self.training = False
        set_keep_awake(False)
        self._save_model_async("停止")
        self.w_view = None
        self.btn_main.config(text="开始训练")
        self.btn_stop.config(state="disabled")
        self.paused = False
    def _on_close(self):
        if self.training:
            self.training = False
            ai2048.CPP.ai_train_stop()
        for _ in range(100):          # 等可能进行中的异步快照写完，防止模型截断
            if not self._saving:
                break
            time.sleep(0.1)
        self._save_model_sync()
        self._save_settings()
        set_keep_awake(False)         # 显式恢复休眠策略（进程退出亦会自动失效）
        self.destroy()

    def _save_model_sync(self):
        """关闭窗口时的同步保存（进程已停，不存在并发写）。"""
        if self.w_view is None or self._saving:
            return
        try:
            np.savez(MODEL_PATH, w=self.w_view)
        except Exception:
            pass


if __name__ == "__main__":
    TrainGUI().mainloop()
