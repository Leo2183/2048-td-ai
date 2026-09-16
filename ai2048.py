# -*- coding: utf-8 -*-
"""2048 TD 强化学习智能体：N-tuple 特征网络 + afterstate TD(0)。

算法出自 Szubert & Jaśkowski (2014) / Jaśkowski et al. (2015) 的经典方案，
纯 numpy 实现，训练与推理共用同一模块。

棋盘表示
    长度 16 的 uint8 数组（行优先），值 = log2(数字)：0=空、1=2、2=4 … 11=2048。

奖励机制（按用户要求以分数为信号）
    每步奖励 r = 该步合并产生的得分（两个 8 相撞得 16 分）；终局价值 V=0。
    价值网络 V(afterstate) 学习的目标是"从这个落子后局面出发的期望累计得分"，
    策略 argmax(r + V) 即逐步最大化单局总分。

特征网络
    V(u) = Σ_g W[idx_g(u)]，g 遍历 4 个 6 格 pattern × 旋转/镜像对称得到的
    32 个特征组（同一 pattern 的 8 个对称共享一张 16^6 查找表，共 4 张
    ≈ 268MB），idx_g 为组内 6 格值的 4bit 编码。

TD 更新（afterstate TD(0)）
    V(u_t) ← V(u_t) + α·(r_{t+1} + V(u_{t+1}) − V(u_t))，死局时目标为 0。
"""

from pathlib import Path

import numpy as np

DIRS = "LRUD"
MODEL_DIR = Path(__file__).resolve().parent / "models"

# ---------- 移动查表：65536 种 4 格行 → 压缩合并结果 ----------
def _merge(vals):
    out, reward, i = [], 0, 0
    vals = [v for v in vals if v]   # 丢弃空格
    while i < len(vals):
        if i + 1 < len(vals) and vals[i] == vals[i + 1]:
            out.append(vals[i] + 1)
            reward += 1 << (vals[i] + 1)
            i += 2
        else:
            out.append(vals[i])
            i += 1
    out += [0] * (4 - len(out))
    return out, reward


def _build_tables():
    new_t = np.zeros(65536, dtype=np.int64)
    rev_t = np.zeros(65536, dtype=np.int64)
    score_t = np.zeros(65536, dtype=np.int64)
    for code in range(65536):
        vals = [(code >> (4 * (3 - i))) & 0xF for i in range(4)]
        merged, sc = _merge(vals)
        new_t[code] = sum(v << (4 * (3 - i)) for i, v in enumerate(merged))
        score_t[code] = sc
        rev_t[code] = ((code & 0xF) << 12) | ((code & 0xF0) << 4) | \
                      ((code & 0xF00) >> 4) | ((code & 0xF000) >> 12)
    return new_t, rev_t, score_t


NEW_T, REV_T, SCORE_T = _build_tables()
_ROW_W = np.array([4096, 256, 16, 1], dtype=np.int64)   # 移动查表的行打包权重
_DIR_INT = {"L": 0, "R": 1, "U": 2, "D": 3}

# ---------- numba 加速层（环境无 numba 时自动回退纯 numpy） ----------
try:
    from numba import njit
    HAS_NUMBA = True

    @njit(cache=True)
    def _nb_value(w, pos, w6, offset, board):
        total = 0.0
        for g in range(pos.shape[0]):
            idx = offset[g]
            for k in range(6):
                idx += board[pos[g, k]] * w6[k]
            total += w[idx]
        return total

    @njit(cache=True)
    def _nb_update(w, pos, w6, offset, board, delta):
        for g in range(pos.shape[0]):
            idx = offset[g]
            for k in range(6):
                idx += board[pos[g, k]] * w6[k]
            w[idx] += delta

    @njit(cache=True)
    def _nb_move(board, d, new_t, rev_t, score_t, roww):
        """d: 0=L 1=R 2=U 3=D。返回 (新棋盘, 得分, 是否移动)。"""
        out = board.copy()
        total = 0
        moved = False
        horiz = d < 2
        rev = d == 1 or d == 3
        shifts = (12, 8, 4, 0)
        for i in range(4):
            code = 0
            for k in range(4):
                cell = board[i * 4 + k] if horiz else board[k * 4 + i]
                code += cell * roww[k]
            if rev:
                code = rev_t[code]
            total += score_t[code]
            nc = new_t[code]
            if rev:
                nc = rev_t[nc]
            for k in range(4):
                v = (nc >> shifts[k]) & 0xF
                j = i * 4 + k if horiz else k * 4 + i
                if v != board[j]:
                    moved = True
                out[j] = v
        return out, total, moved

except ImportError:
    HAS_NUMBA = False


def move_board(board, d):
    """d ∈ 'LRUD'。返回 (新棋盘, 合并得分, 是否为有效移动)。"""
    if CPP is not None:
        out = np.zeros(16, dtype=np.uint8)
        sc = CPP._c_int()
        moved = CPP.ai_move(CPP._p8(board), _DIR_INT[d], CPP._p8(out), CPP._byref(sc))
        return out, sc.value, bool(moved)
    if HAS_NUMBA:
        out, total, moved = _nb_move(board, _DIR_INT[d], NEW_T, REV_T, SCORE_T, _ROW_W)
        return out, int(total), bool(moved)
    m = board.reshape(4, 4)
    if d in "LR":
        codes = m.astype(np.int64) @ _ROW_W
        if d == "R":
            codes = REV_T[codes]
        scores = SCORE_T[codes]
        new_codes = NEW_T[codes]
        if d == "R":
            new_codes = REV_T[new_codes]
        rows = np.stack([(new_codes >> s) & 0xF for s in (12, 8, 4, 0)], axis=1)
    else:
        rev = d == "D"
        codes = m.T.astype(np.int64) @ _ROW_W
        if rev:
            codes = REV_T[codes]
        scores = SCORE_T[codes]
        new_codes = NEW_T[codes]
        if rev:
            new_codes = REV_T[new_codes]
        cols = np.stack([(new_codes >> s) & 0xF for s in (12, 8, 4, 0)], axis=1)
        rows = cols.T
    nb = rows.astype(np.uint8).flatten()
    return nb, int(scores.sum()), bool(np.any(nb != board))


def spawn(board, rng):
    """在随机空格放入 2（90%）或 4（10%），返回新棋盘。"""
    empty = np.flatnonzero(board == 0)
    board = board.copy()
    board[empty[rng.integers(len(empty))]] = 1 if rng.random() < 0.9 else 2
    return board


def new_board(rng):
    return spawn(spawn(np.zeros(16, dtype=np.uint8), rng), rng)


# ---------- 特征组：8 个 6 格 pattern × 旋转/镜像对称（同 pattern 共享一张表） ----------
# 八者的对称集合互不相交（共 64 组），蛇形/楼梯是 2048 的经典强结构特征
BASE_PATTERNS = [
    (0, 1, 2, 3, 4, 5),        # 行0 + 行1左两格
    (4, 5, 6, 7, 8, 9),        # 行1 + 行2左两格
    (0, 1, 2, 3, 6, 7),        # 行0 + 行1右两格
    (0, 1, 2, 3, 7, 6),        # 蛇形：行0 顺 + 行1 逆
    (0, 1, 2, 4, 5, 6),        # 2x3 块（左上）
    (1, 2, 3, 5, 6, 7),        # 2x3 块（右上）
    (0, 1, 4, 5, 9, 10),       # 楼梯
    (0, 1, 5, 6, 10, 11),      # 斜带
]


def _build_groups():
    m = np.arange(16).reshape(4, 4)
    perms = []
    for k in range(4):
        r = np.rot90(m, k)
        for flip in (False, True):
            perms.append((np.fliplr(r) if flip else r).flatten())
    perms = np.array(perms)   # 变换后棋盘 = 原棋盘[perm]
    groups, pattern_of = [], []
    for pi, cells in enumerate(BASE_PATTERNS):
        seen = set()
        for perm in perms:
            key = tuple(perm[list(cells)].tolist())   # 有序序列：不同排列=不同特征
            if key not in seen:
                seen.add(key)
                groups.append(key)
                pattern_of.append(pi)
    return np.array(groups, dtype=np.int64), np.array(pattern_of, dtype=np.int64)


POS, PATTERN_OF = _build_groups()
N_GROUPS = len(POS)
TABLE = 1 << 24                      # 16^6 = 16,777,216 项/张
N_TABLES = len(BASE_PATTERNS)
_W = (16 ** np.arange(5, -1, -1)).astype(np.int64)   # 6 格特征索引权重
_OFFSET = PATTERN_OF * TABLE


# ---------- C++ 引擎层（最高优先级；缺 dll 时回退 numba → numpy） ----------
def _load_cpp():
    import ctypes
    from ctypes import POINTER, byref, c_double, c_float, c_int, c_int32, c_int64, c_uint8
    dll_path = Path(__file__).resolve().parent / "engine.dll"
    if not dll_path.exists():
        return None
    try:
        dll = ctypes.CDLL(str(dll_path))
        dll.ai_init.argtypes = [POINTER(c_int32), c_int, POINTER(c_int32), POINTER(c_int64),
                                POINTER(c_int32), POINTER(c_int32), POINTER(c_int32)]
        dll.ai_move.argtypes = [POINTER(c_uint8), c_int, POINTER(c_uint8), POINTER(c_int)]
        dll.ai_value.argtypes = [POINTER(c_uint8), POINTER(c_float)]
        dll.ai_value.restype = c_double
        dll.ai_update.argtypes = [POINTER(c_uint8), POINTER(c_float), c_double]
        dll.ai_choose.argtypes = [POINTER(c_uint8), POINTER(c_float), c_int]
        dll.ai_choose.restype = c_int
        dll.ai_train_start.argtypes = [POINTER(c_float), c_int, c_double]
        dll.ai_train_stats.argtypes = [POINTER(c_int64), c_int]
        dll.ai_train_seg_take.argtypes = [c_int, POINTER(c_int64)]
        dll.ai_train_seg_take.restype = c_int64
        dll.ai_train_set_pause.argtypes = [c_int]
        dll.ai_train_set_alpha.argtypes = [c_double]
        dll.ai_train_set_speed.argtypes = [c_double]
        # 常量注入（单一事实源：全部来自本模块定义）
        p32 = lambda a: a.ctypes.data_as(POINTER(c_int32))
        p64 = lambda a: a.ctypes.data_as(POINTER(c_int64))
        dll.ai_init(p32(POS.astype(np.int32)), N_GROUPS, p32(_W.astype(np.int32)),
                    p64(_OFFSET), p32(NEW_T.astype(np.int32)),
                    p32(REV_T.astype(np.int32)), p32(SCORE_T.astype(np.int32)))
        dll._p8 = lambda a: a.ctypes.data_as(POINTER(c_uint8))
        dll._pf = lambda a: a.ctypes.data_as(POINTER(c_float))
        dll._c_int = c_int
        dll._byref = byref
        return dll
    except (OSError, AttributeError):
        return None


CPP = _load_cpp()


class NTupleNet:
    def __init__(self, weights=None):
        self.w = weights if weights is not None else np.zeros(N_TABLES * TABLE, np.float32)

    def idx(self, board):
        """board → 32 个特征表索引"""
        return board[POS].astype(np.int64) @ _W + _OFFSET

    def value(self, board):
        if CPP is not None:
            return CPP.ai_value(CPP._p8(board), CPP._pf(self.w))
        if HAS_NUMBA:
            return _nb_value(self.w, POS, _W, _OFFSET, board)
        return float(self.w[self.idx(board)].sum())

    def update(self, board, delta):
        if CPP is not None:
            CPP.ai_update(CPP._p8(board), CPP._pf(self.w), delta)
        elif HAS_NUMBA:
            _nb_update(self.w, POS, _W, _OFFSET, board, delta)
        else:
            self.w[self.idx(board)] += delta

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, w=self.w)   # 268MB 表压缩太慢，直接存原样

    @classmethod
    def load(cls, path):
        """加载模型；旧版小表（pattern 数较少）自动半迁移到新表前半。"""
        data = np.load(Path(path))
        w = data["w"]
        full = N_TABLES * TABLE
        if w.shape == (full,):
            return cls(w.astype(np.float32, copy=False))
        if w.ndim == 1 and w.shape[0] and w.shape[0] % TABLE == 0 and w.shape[0] < full:
            new = np.zeros(full, np.float32)
            new[:w.shape[0]] = w       # 旧 pattern 表原样迁入，新增表从零学习
            return cls(new)
        raise ValueError(f"模型形状不符: {w.shape}（可能是不兼容的旧版模型）")


class TDAgent:
    def __init__(self, net=None, alpha=0.005):
        self.net = net if net is not None else NTupleNet()
        self.alpha = alpha

    def choose(self, board):
        """贪心一步。返回 (方向, 落子后棋盘, 该步得分)；死局返回 (None, None, 0)。"""
        best_d, best_after, best_r, best_v = None, None, 0, -np.inf
        for d in DIRS:
            nb, r, moved = move_board(board, d)
            if not moved:
                continue
            v = r + self.net.value(nb)
            if v > best_v:
                best_d, best_after, best_r, best_v = d, nb, r, v
        return best_d, best_after, best_r


# ---------- expectimax 搜索：用学到的 V 做叶评估（只用于推理，不参与训练） ----------
def _value(net, board, cache=None):
    """net.value 的记忆化封装：同一盘面返回逐位相同的结果，不影响棋力。"""
    if cache is None:
        return net.value(board)
    key = b"V" + board.tobytes()
    v = cache.get(key)
    if v is None:
        v = net.value(board)
        cache[key] = v
    return v


def _greedy_value(net, board, cache=None):
    """出块后盘面的近似价值：贪心一步的 max(r + V(after))。

    V 是 afterstate（刚移动完）的价值函数，直接评估出块后的 nextstate
    会有系统性分布偏移（实测反而比不搜索更差），故叶子先贪心展开一步
    对齐 TD 学习目标后再评估。
    """
    if cache is not None:
        hit = cache.get(b"G" + board.tobytes())
        if hit is not None:
            return hit
    best = -np.inf
    for d in DIRS:
        nb, r, moved = move_board(board, d)
        if moved:
            v = r + _value(net, nb, cache)
            if v > best:
                best = v
    result = best if best > -np.inf else 0.0
    if cache is not None:
        cache[b"G" + board.tobytes()] = result
    return result


def _chance_value(net, board, plies, cache=None):
    """随机出块节点的期望价值。空格多时忽略 10% 的 4 块分支以控制耗时。"""
    if cache is not None:
        key = b"C" + board.tobytes() + bytes([plies])
        hit = cache.get(key)
        if hit is not None:
            return hit
    empties = np.flatnonzero(board == 0)
    if len(empties) == 0:
        return _greedy_value(net, board, cache)
    p_tile = 1.0 / len(empties)
    branches = ((1, 0.9),) if len(empties) > 3 else ((1, 0.9), (2, 0.1))
    total = 0.0
    for pos in empties:
        for val, p in branches:
            b = board.copy()
            b[pos] = val
            if plies <= 1:
                total += p_tile * p * _greedy_value(net, b, cache)
            else:
                sub_d, sub_v = expectimax_best(net, b, plies - 1, cache)
                total += p_tile * p * (sub_v if sub_d is not None else 0.0)
    if cache is not None:
        cache[key] = total
    return total


def expectimax_best(net, board, plies=2, cache=None):
    """plies 层 expectimax。返回 (最佳方向, 期望价值)；死局返回 (None, 0)。"""
    best_d, best_v = None, -np.inf
    for d in DIRS:
        nb, r, moved = move_board(board, d)
        if not moved:
            continue
        v = r + _chance_value(net, nb, plies, cache)
        if v > best_v:
            best_d, best_v = d, v
    if best_d is None:
        return None, 0.0
    return best_d, best_v


class ExpectimaxAgent:
    """推理用智能体：expectimax 搜索 + TD 网络叶评估。接口与 TDAgent 一致。

    自适应深度：空格多时分支爆炸、搜索收益低，用浅搜；残局空格少、
    分支收缩且容错为零，加深到 3 层（冲 8192/16384 的关键）。
    置换表缓存估值与子树结果（纯记忆化，结果逐位一致），残局大幅提速。
    """

    def __init__(self, net):
        self.net = net
        self._cache = {}

    def choose(self, board):
        if board.max() > 15:
            board = np.minimum(board, 15)   # tile 降级：超出 4bit 编码的大块按 32768 评估
        empties = int((board == 0).sum())
        if empties >= 9:
            plies = 1      # 开局：快
        elif empties >= 5:
            plies = 2      # 中盘
        elif empties >= 3:
            plies = 3      # 残局前段
        else:
            plies = 4      # 终盘极限：C++ 速度下 4 层也只需几毫秒
        if CPP is not None:
            di = CPP.ai_choose(CPP._p8(board), CPP._pf(self.net.w), plies)
            if di < 0:
                return None, None, 0
            d = "LRUD"[di]
            nb, r, _ = move_board(board, d)
            return d, nb, r
        self._cache.clear()   # 每步搜索独立，避免跨步残留与内存膨胀
        d, _ = expectimax_best(self.net, board, plies, self._cache)
        if d is None:
            return None, None, 0
        nb, r, _ = move_board(board, d)
        return d, nb, r


def play_episode(agent, rng, learn=True):
    """自我对弈一局；learn=True 时做 TD 更新。返回统计 dict。"""
    board = new_board(rng)
    prev_after = None
    score = steps = 0
    while True:
        d, after, r = agent.choose(board)
        if d is None:   # 死局：上一个 afterstate 的目标价值为 0
            if learn and prev_after is not None:
                agent.net.update(prev_after, -agent.alpha * agent.net.value(prev_after))
            break
        if learn and prev_after is not None:
            prev_v = agent.net.value(prev_after)
            agent.net.update(prev_after, agent.alpha * (r + agent.net.value(after) - prev_v))
        prev_after = after
        score += r
        steps += 1
        board = spawn(after, rng)
    return {"score": score, "max_tile": 1 << int(board.max()), "steps": steps}
