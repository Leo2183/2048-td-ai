# -*- coding: utf-8 -*-
"""2048 —— 本地窗口小游戏（tkinter，零第三方依赖）

运行:  python game2048.py
操作:  方向键 / WASD 移动    R 重开一局    胜利后按 Enter 继续玩
       点「AI 演示」可让训练好的强化学习 AI 自动玩（需先运行 train2048.py 生成模型）
"""

import json
import math
import random
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

import numpy as np

import ai2048

# ---------- 常量 ----------
SIZE = 4                                    # 4x4 棋盘
CELL = 100                                  # 格子边长 px
GAP = 12                                    # 格子间距 px
PAD = 14                                    # 棋盘内边距 px
BOARD = PAD * 2 + CELL * SIZE + GAP * (SIZE - 1)

FONT = "Microsoft YaHei UI"

BG        = "#faf8ef"   # 窗口背景
BOARD_BG  = "#bbada0"   # 棋盘背景
CELL_BG   = "#cdc1b4"   # 空格背景
DARK      = "#776e65"   # 深色文字
LIGHT     = "#f9f6f2"   # 浅色文字
BUTTON_BG = "#8f7a66"

TILE_STYLE = {
    2:    ("#eee4da", DARK),
    4:    ("#ede0c8", DARK),
    8:    ("#f2b179", LIGHT),
    16:   ("#f59563", LIGHT),
    32:   ("#f67c5f", LIGHT),
    64:   ("#f65e3b", LIGHT),
    128:  ("#edcf72", LIGHT),
    256:  ("#edcc61", LIGHT),
    512:  ("#edc850", LIGHT),
    1024: ("#edc53f", LIGHT),
    2048: ("#edc22e", LIGHT),
}
SUPER = ("#3c3a32", LIGHT)   # 4096 及以上

SAVE_PATH = Path.home() / ".2048.json"


def tile_style(v):
    return TILE_STYLE.get(v, SUPER)


def number_font(v, scale=1.0):
    if v < 10:
        size = 46
    elif v < 100:
        size = 44
    elif v < 1000:
        size = 36
    elif v < 10000:
        size = 26
    else:
        size = 22
    return (FONT, max(9, int(size * scale)), "bold")


def rrect_pts(x, y, s, r):
    """圆角矩形的 12 个控制点（配合 smooth=True 使用）"""
    x2, y2 = x + s, y + s
    return [x + r, y, x2 - r, y, x2, y, x2, y + r, x2, y2 - r, x2, y2,
            x2 - r, y2, x + r, y2, x, y2, x, y2 - r, x, y + r, x, y]


class Tile:
    __slots__ = ("value", "row", "col", "rect", "text", "anim")

    def __init__(self, value, row, col):
        self.value = value
        self.row = row
        self.col = col
        self.rect = None
        self.text = None
        self.anim = None   # 进行中的缩放动画 after id


def compute_move(tiles, direction):
    """按方向计算一步移动。direction ∈ 'L' 'R' 'U' 'D'。

    返回 (moved, moves, merges, gain)：
      moves:  [(tile, 新row, 新col), ...]   所有方块的目标位置
      merges: [(tileA, tileB, 新row, 新col, 合并值), ...]
    只做纯计算，不修改 tiles。
    """
    horiz = direction in "LR"
    forward = direction in "LU"          # 朝 0 方向移动
    key_attr, pos_attr = ("row", "col") if horiz else ("col", "row")

    lines = {}
    for t in tiles:
        lines.setdefault(getattr(t, key_attr), []).append(t)

    moves, merges = [], []
    gain = 0
    moved = False
    for line in lines.values():
        line.sort(key=lambda t: getattr(t, pos_attr), reverse=not forward)
        target = 0 if forward else SIZE - 1
        step = 1 if forward else -1
        i = 0
        while i < len(line):
            a = line[i]
            if i + 1 < len(line) and line[i + 1].value == a.value:
                b = line[i + 1]
                pos = (getattr(a, key_attr), target) if horiz else (target, getattr(a, key_attr))
                moves.append((a, *pos))
                moves.append((b, *pos))
                merges.append((a, b, *pos, a.value * 2))
                gain += a.value * 2
                moved = True
                target += step
                i += 2
            else:
                pos = (getattr(a, key_attr), target) if horiz else (target, getattr(a, key_attr))
                moves.append((a, *pos))
                if getattr(a, pos_attr) != target:
                    moved = True
                target += step
                i += 1
    return moved, moves, merges, gain


class Game2048(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("2048")
        self.configure(bg=BG)
        self.resizable(False, False)

        self.tiles = []
        self.score = 0
        self.best = load_best()
        self.busy = False        # 滑动动画期间锁输入
        self.won = False
        self.state = "playing"   # playing | win | over

        self.ai_on = False       # AI 演示模式
        self.ai_agent = None
        self._ai_job = None      # AI 步进循环的 after id
        self._ai_speed = 0       # 演示速度档: 0=慢 1=快 2=极速（AI 模式中按 S 切换）

        self._build_ui()
        self._bind_keys()
        self.new_game()

    # ---------- 界面 ----------
    def _build_ui(self):
        outer = tk.Frame(self, bg=BG, padx=22, pady=16)
        outer.pack()

        header = tk.Frame(outer, bg=BG)
        header.pack(fill="x")
        tk.Label(header, text="2048", font=(FONT, 46, "bold"),
                 bg=BG, fg=DARK).pack(side="left", padx=(0, 16))

        scores = tk.Frame(header, bg=BG)
        scores.pack(side="right")
        self.score_lbl = self._score_box(scores, "分数")
        self.best_lbl = self._score_box(scores, "最高分")

        row2 = tk.Frame(outer, bg=BG)
        row2.pack(fill="x", pady=(2, 12))
        tk.Label(row2, text="方向键 / WASD 移动 · R 重开一局",
                 font=(FONT, 11), bg=BG, fg=DARK).pack(side="left")
        btns = tk.Frame(row2, bg=BG)
        btns.pack(side="right")
        self.speed_btn = tk.Button(btns, text="速度:慢速", command=self._cycle_ai_speed,
                                   bg=BUTTON_BG, fg=LIGHT, activebackground="#9f8b77",
                                   activeforeground=LIGHT, relief="flat",
                                   font=(FONT, 12, "bold"), padx=12, pady=4,
                                   cursor="hand2", takefocus=0)
        self.speed_btn.pack(side="left", padx=(0, 8))
        self.ai_btn = tk.Button(btns, text="AI 演示", command=self.toggle_ai, bg=BUTTON_BG,
                                fg=LIGHT, activebackground="#9f8b77", activeforeground=LIGHT,
                                relief="flat", font=(FONT, 12, "bold"), padx=14, pady=4,
                                cursor="hand2", takefocus=0)
        self.ai_btn.pack(side="left", padx=(0, 8))
        tk.Button(btns, text="新游戏", command=self.new_game, bg=BUTTON_BG, fg=LIGHT,
                  activebackground="#9f8b77", activeforeground=LIGHT, relief="flat",
                  font=(FONT, 12, "bold"), padx=16, pady=4, cursor="hand2",
                  takefocus=0).pack(side="left")

        self.canvas = tk.Canvas(outer, width=BOARD, height=BOARD,
                                bg=BOARD_BG, highlightthickness=0)
        self.canvas.pack()
        for r in range(SIZE):
            for c in range(SIZE):
                x, y = self.cell_xy(r, c)
                self.canvas.create_polygon(rrect_pts(x, y, CELL, 8),
                                           smooth=True, fill=CELL_BG, outline="")

    def _score_box(self, parent, title):
        box = tk.Frame(parent, bg=BOARD_BG, padx=16, pady=5)
        box.pack(side="left", padx=(0, 8))
        tk.Label(box, text=title, font=(FONT, 10), bg=BOARD_BG, fg="#eee4da").pack()
        val = tk.Label(box, text="0", font=(FONT, 16, "bold"), bg=BOARD_BG, fg=LIGHT)
        val.pack()
        return val

    def _bind_keys(self):
        self.bind("<Key>", self._on_key)

    def _on_key(self, e):
        keymap = {"Left": "L", "Right": "R", "Up": "U", "Down": "D",
                  "a": "L", "d": "R", "w": "U", "s": "D",
                  "A": "L", "D": "R", "W": "U", "S": "D"}
        if e.keysym in ("r", "R"):
            self.new_game()
            return
        if e.keysym in ("s", "S"):
            self._cycle_ai_speed()
            return
        if e.keysym == "Return" and self.state == "win":
            self.resume()
            return
        d = keymap.get(e.keysym)
        if d:
            self.do_move(d)

    # ---------- 绘制 ----------
    def cell_xy(self, row, col):
        return PAD + col * (CELL + GAP), PAD + row * (CELL + GAP)

    def draw_tile(self, t, scale=1.0):
        """在 t.row/t.col 位置以 scale 比例绘制方块"""
        x, y = self.cell_xy(t.row, t.col)
        fill, tf = tile_style(t.value)
        s = CELL * scale
        off = (CELL - s) / 2
        t.rect = self.canvas.create_polygon(rrect_pts(x + off, y + off, s, 8 * max(scale, 0.3)),
                                            smooth=True, fill=fill, outline="")
        if scale > 0.55:
            t.text = self.canvas.create_text(x + CELL / 2, y + CELL / 2,
                                             text=str(t.value),
                                             font=number_font(t.value, scale), fill=tf)
        else:
            t.text = None

    def _erase_tile(self, t):
        self.canvas.delete(t.rect)
        if t.text:
            self.canvas.delete(t.text)
        t.rect = t.text = None

    def draw_tile_at(self, t, rowf, colf):
        """按浮点行列坐标移动已有图形（滑动动画用）"""
        x = PAD + colf * (CELL + GAP)
        y = PAD + rowf * (CELL + GAP)
        self.canvas.coords(t.rect, *rrect_pts(x, y, CELL, 8))
        if t.text:
            self.canvas.coords(t.text, x + CELL / 2, y + CELL / 2)

    def cancel_tile_anim(self, t):
        if t.anim:
            self.after_cancel(t.anim)
            t.anim = None
            self._erase_tile(t)
            self.draw_tile(t)

    # ---------- 动画 ----------
    def _cycle_ai_speed(self):
        names = ("慢速", "快速", "极速")
        self._ai_speed = (self._ai_speed + 1) % 3
        self.speed_btn.config(text=f"速度:{names[self._ai_speed]}")

    def animate_slide(self, anims, on_done):
        """anims: [(tile, 起始row, 起始col)]，目标位置读 tile.row/col"""
        if self.ai_on and self._ai_speed == 2:   # 极速：跳过动画直接落位
            for t, _, _ in anims:
                self.draw_tile_at(t, t.row, t.col)
            on_done()
            return
        steps, delay = (4, 8) if self.ai_on else (8, 15)   # AI 模式用快档动画
        state = {"i": 0}

        def tick():
            state["i"] += 1
            p = state["i"] / steps
            p = 1 - (1 - p) ** 2   # ease-out
            for t, fr, fc in anims:
                self.draw_tile_at(t, fr + (t.row - fr) * p, fc + (t.col - fc) * p)
            if state["i"] < steps:
                self.after(delay, tick)
            else:
                for t, _, _ in anims:
                    self.draw_tile_at(t, t.row, t.col)
                on_done()

        self.after(delay, tick)

    def animate_scale(self, t, mode):
        """mode: 'appear' 从 0 放大 / 'pop' 合并弹跳"""
        if self.ai_on and self._ai_speed == 2:   # 极速：直接完成态
            t.anim = None
            return
        steps, delay = (4, 10) if self.ai_on else (7, 15)
        state = {"i": 0}

        def tick():
            state["i"] += 1
            p = state["i"] / steps
            s = p if mode == "appear" else 1 + 0.18 * math.sin(math.pi * p)
            self._erase_tile(t)
            self.draw_tile(t, scale=s)
            if state["i"] < steps:
                t.anim = self.after(delay, tick)
            else:
                t.anim = None

        t.anim = self.after(delay, tick)

    # ---------- 游戏流程 ----------
    def new_game(self):
        for t in self.tiles:
            self.cancel_tile_anim(t)
            self._erase_tile(t)
        self.canvas.delete("overlay")
        self.tiles = []
        self.score = 0
        self.won = False
        self.state = "playing"
        self.busy = False
        self.update_scores()
        self.spawn_random()
        self.spawn_random()
        if self.ai_on:   # AI 演示中重开局则重启自动走棋循环
            if self._ai_job:
                self.after_cancel(self._ai_job)
            self._ai_job = self.after(250, self.ai_step)

    def spawn_random(self):
        empty = [(r, c) for r in range(SIZE) for c in range(SIZE)
                 if all(t.row != r or t.col != c for t in self.tiles)]
        if not empty:
            return
        r, c = random.choice(empty)
        t = Tile(random.choices((2, 4), (0.9, 0.1))[0], r, c)
        self.tiles.append(t)
        self.draw_tile(t, scale=1.0)
        self.animate_scale(t, "appear")

    def do_move(self, direction):
        if self.busy or self.state != "playing":
            return
        moved, moves, merges, gain = compute_move(self.tiles, direction)
        if not moved:
            return
        self.busy = True
        self.score += gain
        if self.score > self.best:
            self.best = self.score
            save_best(self.best)
        self.update_scores()

        merged_ids = set()
        for a, b, *_ in merges:
            merged_ids.update((id(a), id(b)))
        anims = []
        for t, nr, nc in moves:
            self.cancel_tile_anim(t)
            anims.append((t, t.row, t.col))
            t.row, t.col = nr, nc
        # 被合并的方块立即退出逻辑盘面（图形保留到滑动结束再删）
        self.tiles = [t for t in self.tiles if id(t) not in merged_ids]
        merge_spawns = [(nr, nc, v) for _, _, nr, nc, v in merges]

        def on_slide_done():
            for a, b, *_ in merges:
                self._erase_tile(a)
                self._erase_tile(b)
            for nr, nc, v in merge_spawns:
                t = Tile(v, nr, nc)
                self.tiles.append(t)
                self.draw_tile(t)
                self.animate_scale(t, "pop")
            self.spawn_random()

            self.busy = False
            if not self.won and any(v >= 2048 for _, _, v in merge_spawns):
                self.won = True
                self.state = "win"
                self.show_win()
            elif not self.has_moves():
                self.state = "over"
                self.show_over()

        self.animate_slide(anims, on_slide_done)

    def has_moves(self):
        if len(self.tiles) < SIZE * SIZE:
            return True
        grid = {(t.row, t.col): t.value for t in self.tiles}
        for r in range(SIZE):
            for c in range(SIZE):
                if c + 1 < SIZE and grid[(r, c)] == grid[(r, c + 1)]:
                    return True
                if r + 1 < SIZE and grid[(r, c)] == grid[(r + 1, c)]:
                    return True
        return False

    def update_scores(self):
        self.score_lbl.config(text=str(self.score))
        self.best_lbl.config(text=str(self.best))

    def resume(self):
        self.canvas.delete("overlay")
        self.state = "playing"

    # ---------- AI 演示 ----------
    def toggle_ai(self):
        if not self.ai_on:
            try:
                self.ai_agent = ai2048.ExpectimaxAgent(
                    ai2048.NTupleNet.load(ai2048.MODEL_DIR / "agent.npz"))
            except Exception as e:
                messagebox.showwarning(
                    "AI 演示", f"加载模型失败：{e}\n\n请先运行  python train2048.py  训练")
                return
            self.ai_on = True
            # 深搜线程是 GIL 密集计算，缩短线程切换间隔让 UI 动画能插进来
            sys.setswitchinterval(0.001)
            self.ai_btn.config(text="停止 AI")
            self.new_game()
        else:
            self.ai_on = False
            sys.setswitchinterval(0.005)   # 恢复 CPython 默认
            self.ai_btn.config(text="AI 演示")
            if self._ai_job:
                self.after_cancel(self._ai_job)
                self._ai_job = None

    def ai_step(self):
        self._ai_job = None
        if not self.ai_on:
            return
        if self.state == "win":    # AI 模式自动跳过胜利确认
            self.resume()
        if self.state != "playing":
            return
        if self.busy:              # 上一步动画没走完，稍后重试
            self._ai_job = self.after((40, 20, 5)[self._ai_speed], self.ai_step)
            return
        board = np.zeros(16, dtype=np.uint8)
        for t in self.tiles:
            board[t.row * 4 + t.col] = round(math.log2(t.value))

        def think():               # expectimax 搜索放后台线程，避免卡住动画
            d, _, _ = self.ai_agent.choose(board.copy())
            self.after(0, lambda: self._ai_apply(d))

        threading.Thread(target=think, daemon=True).start()

    def _ai_apply(self, d):
        if not self.ai_on or d is None or self.busy or self.state != "playing":
            return
        self.do_move(d)
        # 间隔按速度档：动画走完再启动下一轮深搜，减少 GIL 争抢窗口
        self._ai_job = self.after((90, 30, 8)[self._ai_speed], self.ai_step)

    def _ai_auto_restart(self):
        if self.ai_on and self.state == "over":
            self.new_game()

    # ---------- 覆盖层 ----------
    def _overlay_bg(self):
        self.canvas.create_rectangle(0, 0, BOARD, BOARD, fill="#eee4da",
                                     outline="", tags="overlay")

    def _overlay_button(self, cx, cy, w, text, tag, cmd):
        self.canvas.create_rectangle(cx - w / 2, cy - 22, cx + w / 2, cy + 22,
                                     fill=BUTTON_BG, outline="", tags=("overlay", tag))
        self.canvas.create_text(cx, cy, text=text, font=(FONT, 15, "bold"),
                                fill=LIGHT, tags=("overlay", tag))
        self.canvas.tag_bind(tag, "<Button-1>", lambda e: cmd())

    def show_win(self):
        self._overlay_bg()
        self.canvas.create_text(BOARD / 2, BOARD / 2 - 90, text="你合成了 2048！",
                                font=(FONT, 34, "bold"), fill=DARK, tags="overlay")
        self.canvas.create_text(BOARD / 2, BOARD / 2 - 50, text=f"得分 {self.score}",
                                font=(FONT, 14), fill=DARK, tags="overlay")
        self._overlay_button(BOARD / 2 - 85, BOARD / 2 + 5, 150, "继续游戏",
                             "ov-continue", self.resume)
        self._overlay_button(BOARD / 2 + 85, BOARD / 2 + 5, 150, "新开一局",
                             "ov-restart", self.new_game)
        self.canvas.create_text(BOARD / 2, BOARD / 2 + 70, text="Enter 也可以继续",
                                font=(FONT, 11), fill=DARK, tags="overlay")

    def show_over(self):
        self._overlay_bg()
        self.canvas.create_text(BOARD / 2, BOARD / 2 - 60, text="游戏结束",
                                font=(FONT, 38, "bold"), fill=DARK, tags="overlay")
        self.canvas.create_text(BOARD / 2, BOARD / 2 - 12, text=f"得分 {self.score}",
                                font=(FONT, 15), fill=DARK, tags="overlay")
        self._overlay_button(BOARD / 2, BOARD / 2 + 55, 170, "再来一局",
                             "ov-restart", self.new_game)
        if self.ai_on:   # AI 演示中 2.5 秒后自动重开
            self.after(2500, self._ai_auto_restart)


def load_best():
    try:
        return int(json.loads(SAVE_PATH.read_text("utf-8"))["best"])
    except Exception:
        return 0


def save_best(v):
    try:
        SAVE_PATH.write_text(json.dumps({"best": v}), encoding="utf-8")
    except Exception:
        pass


if __name__ == "__main__":
    Game2048().mainloop()
