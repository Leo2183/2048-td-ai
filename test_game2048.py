# -*- coding: utf-8 -*-
"""game2048 的逻辑测试 + 窗口冒烟测试（自动退出，不开残留窗口）"""
import game2048 as g


def mk(vals):
    """{(row,col): value} -> [Tile]"""
    return [g.Tile(v, r, c) for (r, c), v in vals.items()]


def final_board(moves, merges):
    """移动结束后的盘面（被合并方块取合并值）"""
    board = {}
    dead = {id(a) for m in merges for a in m[:2]}
    for t, nr, nc in moves:
        board[(nr, nc)] = t.value
    for a, b, nr, nc, v in merges:
        board[(nr, nc)] = v
        assert id(a) in dead and id(b) in dead
    return board


def test_left_merge():
    # [_,2,2,4] 左移 -> [4,4,_,_]
    tiles = mk({(0, 1): 2, (0, 2): 2, (0, 3): 4})
    moved, moves, merges, gain = g.compute_move(tiles, "L")
    assert moved and gain == 4 and len(merges) == 1
    assert final_board(moves, merges)[(0, 0)] == 4
    assert final_board(moves, merges)[(0, 1)] == 4
    assert len(final_board(moves, merges)) == 2


def test_double_pair():
    # [2,2,2,2] 左移 -> [4,4]
    tiles = mk({(0, 0): 2, (0, 1): 2, (0, 2): 2, (0, 3): 2})
    moved, moves, merges, gain = g.compute_move(tiles, "L")
    assert gain == 8 and len(merges) == 2
    b = final_board(moves, merges)
    assert b == {(0, 0): 4, (0, 1): 4}


def test_triple_prefers_near_side():
    # [2,2,4] 左移 -> [4,4]（靠左的一对先合并）
    tiles = mk({(0, 0): 2, (0, 1): 2, (0, 2): 4})
    _, moves, merges, gain = g.compute_move(tiles, "L")
    b = final_board(moves, merges)
    assert b == {(0, 0): 4, (0, 1): 4} and gain == 4


def test_no_move():
    # 全部贴在顶行且值互不相同 -> 上移无效
    tiles = mk({(0, 0): 2, (0, 1): 4, (0, 3): 8})
    moved, moves, merges, gain = g.compute_move(tiles, "U")
    assert not moved and not merges and gain == 0


def test_right_and_down():
    # [2,2,_,_] 右移 -> [_,_,_,4]
    tiles = mk({(0, 0): 2, (0, 1): 2})
    moved, moves, merges, gain = g.compute_move(tiles, "R")
    assert gain == 4 and final_board(moves, merges) == {(0, 3): 4}
    # 竖列 (0,0)=4,(2,0)=4 下移 -> (3,0)=8
    tiles = mk({(0, 0): 4, (2, 0): 4})
    moved, moves, merges, gain = g.compute_move(tiles, "D")
    assert gain == 8 and final_board(moves, merges) == {(3, 0): 8}


def test_merge_across_gap():
    # [2,_,_,2] 左移 -> [4,_,_,_]
    tiles = mk({(0, 0): 2, (0, 3): 2})
    moved, moves, merges, gain = g.compute_move(tiles, "L")
    assert moved and gain == 4 and final_board(moves, merges) == {(0, 0): 4}


def test_smoke_window():
    app = g.Game2048()
    app.update()
    assert len(app.tiles) == 2 and app.score == 0
    # 模拟一串按键，跑完动画后自动退出
    for key in ("<Left>", "<Up>", "<Right>", "<Down>", "<Left>", "<Down>"):
        app.event_generate(key)
        app.update_idletasks()
        app.update()
    app.after(600, app.destroy)
    app.mainloop()
    print("window smoke ok")


if __name__ == "__main__":
    test_left_merge()
    test_double_pair()
    test_triple_prefers_near_side()
    test_no_move()
    test_right_and_down()
    test_merge_across_gap()
    print("logic tests ok")
    test_smoke_window()
    print("all tests passed")
