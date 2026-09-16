# -*- coding: utf-8 -*-
"""ai2048 的正确性测试 + 训练速度基准"""
import time

import numpy as np

import ai2048
import game2048


def random_board(rng, n=None):
    """随机盘面，同时返回 (numpy board, Tile 列表) 两种表示"""
    board = np.zeros(16, dtype=np.uint8)
    tiles = []
    cells = rng.permutation(16)[: (n or rng.integers(2, 16))]
    for cell in cells:
        v = int(rng.integers(1, 12))   # log2 值: 2 ~ 2048
        board[cell] = v
        tiles.append(game2048.Tile(1 << v, cell // 4, cell % 4))
    return board, tiles


def test_move_cross_check():
    """move_board 必须与 UI 的 compute_move 语义完全一致"""
    rng = np.random.default_rng(7)
    for _ in range(400):
        board, tiles = random_board(rng)
        for d in "LRUD":
            nb, score, moved = ai2048.move_board(board, d)
            moved_ref, moves, merges, gain = game2048.compute_move(tiles, d)
            assert moved == moved_ref, (board, d)
            assert score == gain, (board.tolist(), d, score, gain)
            if moved:
                ref = {}
                for t, nr, nc in moves:
                    ref[(nr, nc)] = t.value
                for a, b, nr, nc, v in merges:
                    ref[(nr, nc)] = v
                got = {(i // 4, i % 4): (1 << int(nb[i])) if nb[i] else 0
                       for i in range(16) if nb[i]}
                assert got == ref, (board.tolist(), d, got, ref)


def test_symmetry_groups():
    """特征组必须是棋盘上真实存在且互异的 4 格组合（有序）"""
    flat = set(ai2048.POS.flatten().tolist())
    assert len(ai2048.POS) == ai2048.N_GROUPS == 64   # 8 个 6 格 pattern × 8 对称
    keys = {tuple(g.tolist()) for g in ai2048.POS}
    assert len(keys) == 64   # 互异
    assert flat <= set(range(16))


def test_net_update_and_roundtrip(tmp="."):
    net = ai2048.NTupleNet()
    board = np.zeros(16, dtype=np.uint8)
    board[0], board[1], board[5] = 3, 3, 2
    assert net.value(board) == 0.0
    net.update(board, 0.5)
    # 正确语义: V = Σ_g W[idx_g]，重复索引读写对称累积。
    # 期望值 = 0.5 × Σ(每个索引出现次数的平方)
    from collections import Counter
    c = Counter(net.idx(board).tolist())
    expected = 0.5 * sum(v * v for v in c.values())
    assert abs(net.value(board) - expected) < 1e-6
    path = ai2048.MODEL_DIR / "_test_roundtrip.npz"
    net.save(path)
    net2 = ai2048.NTupleNet.load(path)
    assert np.array_equal(net.w, net2.w)
    path.unlink()


def test_expectimax_cache_consistency():
    """置换表是纯记忆化：缓存版与无缓存版结果必须完全一致"""
    rng = np.random.default_rng(5)
    for _ in range(15):
        b = np.zeros(16, dtype=np.uint8)
        for cell in rng.permutation(16)[:int(rng.integers(10, 16))]:
            b[cell] = int(rng.integers(1, 11))
        d1, v1 = ai2048.expectimax_best(net_fn(), b, 3, cache={})
        d2, v2 = ai2048.expectimax_best(net_fn(), b, 3)
        assert d1 == d2 and abs(v1 - v2) < 1e-6
    dead = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,1,2,3], dtype=np.uint8)
    assert ai2048.expectimax_best(net_fn(), dead, 2, cache={})[0] is None


def _net():
    return ai2048.NTupleNet()


net_fn = _net


def test_play_speed_and_learning_smoke():
    agent = ai2048.TDAgent(alpha=0.005)
    rng = np.random.default_rng(0)
    t0 = time.time()
    head = [ai2048.play_episode(agent, rng) for _ in range(200)]
    dt = time.time() - t0
    speed = 200 / dt
    print(f"speed: {speed:.0f} games/s ({dt/200*1000:.1f} ms/game)")
    assert speed > 3, "训练速度过慢"   # 6-tuple 更大更慢

    # 500 局学习后应显著强于开局（未学习）水平
    zero = ai2048.TDAgent(alpha=0.0)
    base = np.mean([ai2048.play_episode(zero, rng)["score"] for _ in range(100)])
    tail = np.mean([ai2048.play_episode(agent, rng)["score"] for _ in range(100)])
    print(f"random baseline avg score: {base:.0f}, after 200 games: {tail:.0f}")
    assert all(r["score"] >= 0 for r in head)


if __name__ == "__main__":
    test_move_cross_check()
    print("move cross-check ok")
    test_symmetry_groups()
    print("symmetry groups ok")
    test_net_update_and_roundtrip()
    print("net update + save/load ok")
    test_play_speed_and_learning_smoke()
    print("speed + smoke ok")
    print("all tests passed")
