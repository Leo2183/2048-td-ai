# -*- coding: utf-8 -*-
"""2048 TD 强化学习训练器 —— 自我对弈 + 分数奖励迭代。

用法示例:
    python train2048.py --time 300          # 训练 5 分钟（默认）
    python train2048.py --games 10000       # 训 1 万局
    python train2048.py --time 600 --alpha 0.002   # 更长训练用更小步长
    再次运行自动加载 models/agent.npz 继续变强（续训）。

模型保存在 models/agent.npz，训练历史追加在 models/history.jsonl。
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import ai2048

MODEL_PATH = ai2048.MODEL_DIR / "agent.npz"
HISTORY_PATH = ai2048.MODEL_DIR / "history.jsonl"


def load_trained_games():
    """从历史文件里数出累计训练局数"""
    n = 0
    if HISTORY_PATH.exists():
        for line in HISTORY_PATH.read_text("utf-8").splitlines():
            try:
                n += json.loads(line)["count"]
            except (ValueError, KeyError):
                pass
    return n


def main():
    ap = argparse.ArgumentParser(description="2048 TD 强化学习训练器")
    ap.add_argument("--time", type=float, default=300, help="训练秒数（0 表示不限）")
    ap.add_argument("--games", type=int, default=0, help="训练局数上限（0 表示不限）")
    ap.add_argument("--alpha", type=float, default=0.005, help="TD 学习步长")
    ap.add_argument("--report", type=int, default=500, help="每多少局输出一次统计")
    ap.add_argument("--model", type=Path, default=MODEL_PATH, help="模型路径")
    args = ap.parse_args()
    if args.time <= 0 and args.games <= 0:
        ap.error("--time 与 --games 至少给一个大于 0 的值")

    rng = np.random.default_rng()
    if args.model.exists():
        agent = ai2048.TDAgent(ai2048.NTupleNet.load(args.model), alpha=args.alpha)
        print(f"已加载模型 {args.model}，继续训练")
    else:
        agent = ai2048.TDAgent(alpha=args.alpha)
        print("从零开始训练")
    total_before = load_trained_games()
    print(f"累计已训练 {total_before} 局 | 步长 alpha={args.alpha} | Ctrl+C 可随时中断，"
          f"进度自动保存\n")

    window, total_done = [], 0
    t0 = t_window_start = time.time()

    def report():
        scores = [r["score"] for r in window]
        tiles = [r["max_tile"] for r in window]
        elapsed = max(time.time() - t_window_start, 1e-9)
        print(f"[{total_before + total_done - len(window):>7,}"
              f"-{total_before + total_done:>7,}] "
              f"平均 {np.mean(scores):>7.0f} | 最佳 {max(scores):>6,} | "
              f"≥1024 {sum(t >= 1024 for t in tiles) / len(tiles):>5.1%} | "
              f"≥2048 {sum(t >= 2048 for t in tiles) / len(tiles):>5.1%} | "
              f"≥4096 {sum(t >= 4096 for t in tiles) / len(tiles):>5.1%} | "
              f"≥8192 {sum(t >= 8192 for t in tiles) / len(tiles):>5.1%} | "
              f"最大块 {max(tiles):>6,} | {len(window) / elapsed:.0f} 局/秒")
        record = {"games": total_before + total_done, "count": len(window),
                  "avg_score": float(np.mean(scores)), "best_score": max(scores),
                  "max_tile": max(tiles),
                  "rate_1024": sum(t >= 1024 for t in tiles) / len(tiles),
                  "rate_2048": sum(t >= 2048 for t in tiles) / len(tiles),
                  "rate_4096": sum(t >= 4096 for t in tiles) / len(tiles),
                  "rate_8192": sum(t >= 8192 for t in tiles) / len(tiles)}
        with open(HISTORY_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    try:
        while (args.games <= 0 or total_done < args.games) and \
              (args.time <= 0 or time.time() - t0 < args.time):
            window.append(ai2048.play_episode(agent, rng))
            total_done += 1
            if len(window) == args.report:
                report()
                window.clear()
                t_window_start = time.time()
    except KeyboardInterrupt:
        print("\n收到中断信号")
    finally:
        if window:
            report()   # 结束时把不满一个窗口的局也记上
        agent.net.save(args.model)
        mins = (time.time() - t0) / 60
        print(f"\n本次训练 {total_done:,} 局，用时 {mins:.1f} 分钟")
        print(f"累计 {total_before + total_done:,} 局 | 模型已保存到 {args.model}")


if __name__ == "__main__":
    main()
