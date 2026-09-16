// engine.cpp — 2048 TD/expectimax C++ 引擎（ctypes 接口，零第三方依赖）
//
// 设计约定：
//   - 权重表与特征常量（POS/W6/OFFSET/行查表）全部由 Python 侧 numpy 数组持有，
//     通过 ai_init() 传入指针，保证与 Python/numba 实现单一事实源、逐位一致。
//   - 更新语义：逐特征组累加（读写对称，与 numba 版一致）。
//   - expectimax 含贪心叶与置换表，结构与 Python 版逐分支对齐。

#include <atomic>
#include <cstring>
#include <cstdint>
#include <mutex>
#include <memory>
#include <random>
#include <thread>
#include <unordered_map>
#include <vector>

// ---------------- 常量（ai_init 注入） ----------------
static int32_t g_pos[64][6];
static int32_t g_w6[6];
static int64_t g_offset[64];
static int32_t g_new_t[65536], g_rev_t[65536], g_score_t[65536];
static int g_n_groups = 0;

static inline double value_of(const uint8_t* b, const float* w) {
    double total = 0.0;
    for (int g = 0; g < g_n_groups; g++) {
        int64_t idx = g_offset[g];
        const int32_t* p = g_pos[g];
        for (int k = 0; k < 6; k++) idx += (int64_t)b[p[k]] * g_w6[k];
        total += w[idx];
    }
    return total;
}

static inline void update_at(const uint8_t* b, float* w, double delta) {
    for (int g = 0; g < g_n_groups; g++) {
        int64_t idx = g_offset[g];
        const int32_t* p = g_pos[g];
        for (int k = 0; k < 6; k++) idx += (int64_t)b[p[k]] * g_w6[k];
        w[idx] += (float)delta;
    }
}

// ---------------- 移动 ----------------
static inline int move_b(const uint8_t* board, int d, uint8_t* out, int* score) {
    int total = 0;
    bool moved = false;
    const bool horiz = d < 2;
    const bool rev = (d == 1 || d == 3);
    const int shifts[4] = {12, 8, 4, 0};
    const int roww[4] = {4096, 256, 16, 1};
    for (int i = 0; i < 4; i++) {
        int code = 0;
        for (int k = 0; k < 4; k++)
            code += (horiz ? board[i * 4 + k] : board[k * 4 + i]) * roww[k];
        if (rev) code = g_rev_t[code];
        total += g_score_t[code];
        int nc = g_new_t[code];
        if (rev) nc = g_rev_t[nc];
        for (int k = 0; k < 4; k++) {
            int v = (nc >> shifts[k]) & 0xF;
            int j = horiz ? i * 4 + k : k * 4 + i;
            if (v != board[j]) moved = true;
            out[j] = (uint8_t)v;
        }
    }
    *score = total;
    return moved ? 1 : 0;
}

// ---------------- expectimax（贪心叶 + 置换表） ----------------
static inline uint64_t pack16(const uint8_t* b) {
    uint64_t k = 0;
    for (int i = 0; i < 16; i++) k |= ((uint64_t)b[i]) << (4 * i);
    return k;
}

typedef std::unordered_map<uint64_t, double> TT;

static double greedy_value(const uint8_t* b, const float* w, TT& tt) {
    const uint64_t key = pack16(b) * 8 + 6;   // G 槽
    auto it = tt.find(key);
    if (it != tt.end()) return it->second;
    double best = -1e18;
    for (int d = 0; d < 4; d++) {
        uint8_t nb[16];
        int sc;
        if (move_b(b, d, nb, &sc)) {
            double v = sc + value_of(nb, w);
            if (v > best) best = v;
        }
    }
    double r = (best <= -1e17) ? 0.0 : best;
    tt.emplace(key, r);
    return r;
}

static double chance_value(const uint8_t* b, const float* w, int plies, TT& tt) {
    const uint64_t key = pack16(b) * 8 + (uint64_t)plies;   // C 槽（plies 1..3）
    auto it = tt.find(key);
    if (it != tt.end()) return it->second;
    int empties[16], ne = 0;
    for (int i = 0; i < 16; i++) if (!b[i]) empties[ne++] = i;
    double total;
    if (ne == 0) {
        total = greedy_value(b, w, tt);
    } else {
        const double p = 1.0 / ne;
        const int branches = (ne > 3) ? 1 : 2;
        total = 0.0;
        for (int e = 0; e < ne; e++) {
            for (int bi = 0; bi < branches; bi++) {
                const int val = (bi == 0) ? 1 : 2;
                const double pv = (bi == 0) ? 0.9 : 0.1;
                uint8_t nb[16];
                memcpy(nb, b, 16);
                nb[empties[e]] = (uint8_t)val;
                if (plies <= 1) {
                    total += p * pv * greedy_value(nb, w, tt);
                } else {
                    int bd = -1;
                    double bv = -1e18;
                    for (int d = 0; d < 4; d++) {
                        uint8_t mb[16];
                        int sc;
                        if (move_b(nb, d, mb, &sc)) {
                            double v = sc + chance_value(mb, w, plies - 1, tt);
                            if (v > bv) { bv = v; bd = d; }
                        }
                    }
                    total += p * pv * ((bd < 0) ? 0.0 : bv);
                }
            }
        }
    }
    tt.emplace(key, total);
    return total;
}

// ---------------- 训练（多线程 Hogwild） ----------------
struct Worker {
    std::thread th;
    // 滚动窗口（实时卡片）
    int64_t win_score[250];
    int64_t win_tile[250];
    int win_head = 0, win_n = 0;
    int64_t win_sum = 0, win_2048 = 0, win_4096 = 0, win_8192 = 0, win_16384 = 0;
    // 段累计（每满 250 局由 Python 取走写历史）
    std::mutex mtx;
    int64_t seg_n = 0, seg_sum = 0, seg_best = 0, seg_max = 0;
    int64_t seg_2048 = 0, seg_4096 = 0, seg_8192 = 0, seg_1024 = 0, seg_16384 = 0;
    // 会话极值
    int64_t total = 0, best = 0, max_tile = 0;
};

static std::atomic<bool> g_stop{false};
static std::atomic<bool> g_pause{false};
static std::atomic<double> g_alpha{0.005};
static std::atomic<double> g_speed{0.0};   // 每 worker 局/秒配额，0 不限
static std::vector<std::unique_ptr<Worker>> g_workers;
static const float* g_w = nullptr;

static void spawn_tile(uint8_t* b, std::mt19937_64& rng) {
    int empties[16], ne = 0;
    for (int i = 0; i < 16; i++) if (!b[i]) empties[ne++] = i;
    if (!ne) return;
    double u = (rng() >> 11) * (1.0 / 9007199254740992.0);
    b[empties[(int)(u * ne) % ne]] = (u < 0.9) ? 1 : 2;
}

static void worker_loop(int wid) {
    Worker& W = *g_workers[wid];
    std::mt19937_64 rng((uint64_t)(std::random_device{}()) * 0x9E3779B97F4A7C15ULL + wid);
    float* w = const_cast<float*>(g_w);
    int64_t batch = 0;
    auto batch_t0 = std::chrono::steady_clock::now();

    while (!g_stop.load(std::memory_order_relaxed)) {
        if (g_pause.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        // ---- 一局（语义对齐 Python play_episode：读写对称 TD(0)）----
        uint8_t board[16] = {0};
        spawn_tile(board, rng);
        spawn_tile(board, rng);
        uint8_t prev_after[16];
        bool has_prev = false;
        int64_t score = 0;
        while (true) {
            int best_d = -1, best_r = 0;
            uint8_t best_after[16];
            double best_v = -1e18;
            for (int d = 0; d < 4; d++) {
                uint8_t nb[16];
                int sc;
                if (move_b(board, d, nb, &sc)) {
                    double v = sc + value_of(nb, w);
                    if (v > best_v) { best_v = v; best_d = d; best_r = sc; memcpy(best_after, nb, 16); }
                }
            }
            if (best_d < 0) {
                if (has_prev) {
                    double pv = value_of(prev_after, w);
                    update_at(prev_after, w, -g_alpha.load() * pv);
                }
                break;
            }
            if (has_prev) {
                double pv = value_of(prev_after, w);
                update_at(prev_after, w, g_alpha.load() * (best_r + value_of(best_after, w) - pv));
            }
            memcpy(prev_after, best_after, 16);
            has_prev = true;
            score += best_r;
            memcpy(board, best_after, 16);
            spawn_tile(board, rng);
        }
        int64_t max_tile = 1;
        for (int i = 0; i < 16; i++) {
            if (board[i] > 0) {
                int64_t t = ((int64_t)1) << board[i];
                if (t > max_tile) max_tile = t;
            }
        }
        // ---- 统计：滚动窗口 + 段累计 + 会话极值 ----
        if (W.win_n == 250) {
            int64_t os = W.win_score[W.win_head], ot = W.win_tile[W.win_head];
            W.win_sum -= os;
            if (ot >= 2048) W.win_2048--;
            if (ot >= 4096) W.win_4096--;
            if (ot >= 8192) W.win_8192--;
            if (ot >= 16384) W.win_16384--;
        } else {
            W.win_n++;
        }
        W.win_score[W.win_head] = score;
        W.win_tile[W.win_head] = max_tile;
        W.win_sum += score;
        if (max_tile >= 2048) W.win_2048++;
        if (max_tile >= 4096) W.win_4096++;
        if (max_tile >= 8192) W.win_8192++;
        if (max_tile >= 16384) W.win_16384++;
        W.win_head = (W.win_head + 1) % 250;
        {
            std::lock_guard<std::mutex> lk(W.mtx);
            W.seg_n++; W.seg_sum += score;
            if (score > W.seg_best) W.seg_best = score;
            if (max_tile > W.seg_max) W.seg_max = max_tile;
            if (max_tile >= 1024) W.seg_1024++;
            if (max_tile >= 2048) W.seg_2048++;
            if (max_tile >= 4096) W.seg_4096++;
            if (max_tile >= 8192) W.seg_8192++;
            if (max_tile >= 16384) W.seg_16384++;
        }
        W.total++;
        if (score > W.best) W.best = score;
        if (max_tile > W.max_tile) W.max_tile = max_tile;
        // ---- 限速：每 10 局校准一次配额 ----
        double sl = g_speed.load();
        if (sl > 0) {
            batch++;
            if (batch >= 10) {
                double spent = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - batch_t0).count();
                double lag = batch / sl - spent;
                if (lag > 0) std::this_thread::sleep_for(std::chrono::duration<double>(lag));
                batch = 0;
                batch_t0 = std::chrono::steady_clock::now();
            }
        }
    }
}

// ---------------- C 接口 ----------------
extern "C" {

__declspec(dllexport) void ai_init(
    const int32_t* pos, int n_groups, const int32_t* w6, const int64_t* offset,
    const int32_t* new_t, const int32_t* rev_t, const int32_t* score_t)
{
    g_n_groups = n_groups;
    memcpy(g_pos, pos, sizeof(int32_t) * 6 * n_groups);
    memcpy(g_w6, w6, sizeof(int32_t) * 6);
    memcpy(g_offset, offset, sizeof(int64_t) * n_groups);
    memcpy(g_new_t, new_t, 4 * 65536);
    memcpy(g_rev_t, rev_t, 4 * 65536);
    memcpy(g_score_t, score_t, 4 * 65536);
}

__declspec(dllexport) int ai_move(const uint8_t* board, int d, uint8_t* out, int* score) {
    return move_b(board, d, out, score);
}

__declspec(dllexport) double ai_value(const uint8_t* board, const float* w) {
    return value_of(board, w);
}

__declspec(dllexport) void ai_update(const uint8_t* board, float* w, double delta) {
    update_at(board, w, delta);
}

__declspec(dllexport) int ai_choose(const uint8_t* board, const float* w, int plies) {
    uint8_t b[16];
    memcpy(b, board, 16);
    for (int i = 0; i < 16; i++) if (b[i] > 15) b[i] = 15;   // tile 降级防御
    TT tt;
    tt.reserve(4096);
    int best_d = -1;
    double best_v = -1e18;
    for (int d = 0; d < 4; d++) {
        uint8_t nb[16];
        int sc;
        if (!move_b(b, d, nb, &sc)) continue;
        double v = sc + chance_value(nb, w, plies, tt);
        if (v > best_v) { best_v = v; best_d = d; }
    }
    return best_d;
}

__declspec(dllexport) int ai_train_start(float* w, int n_threads, double alpha) {
    if (!g_workers.empty()) return 0;   // 已在训练
    g_stop = false;
    g_pause = false;
    g_alpha = alpha;
    g_w = w;
    g_workers.resize(n_threads);
    for (int i = 0; i < n_threads; i++) {
        g_workers[i].reset(new Worker());
        g_workers[i]->th = std::thread(worker_loop, i);
    }
    return n_threads;
}

__declspec(dllexport) void ai_train_stop() {
    g_stop = true;
    for (auto& W : g_workers)
        if (W->th.joinable()) W->th.join();
    g_workers.clear();
}

__declspec(dllexport) void ai_train_set_pause(int p) { g_pause = (p != 0); }
__declspec(dllexport) void ai_train_set_alpha(double a) { g_alpha = a; }
__declspec(dllexport) void ai_train_set_speed(double s) { g_speed = s; }

// 输出每 worker 10 个 int64:
// total, best, max_tile, win_n, win_sum, win_2048, win_4096, win_8192, seg_n, win_16384
__declspec(dllexport) void ai_train_stats(int64_t* out, int max_workers) {
    int n = (int)g_workers.size();
    if (n > max_workers) n = max_workers;
    for (int i = 0; i < n; i++) {
        Worker& W = *g_workers[i];
        out[i * 10 + 0] = W.total;
        out[i * 10 + 1] = W.best;
        out[i * 10 + 2] = W.max_tile;
        out[i * 10 + 3] = W.win_n;
        out[i * 10 + 4] = W.win_sum;
        out[i * 10 + 5] = W.win_2048;
        out[i * 10 + 6] = W.win_4096;
        out[i * 10 + 7] = W.win_8192;
        out[i * 10 + 8] = W.seg_n;
        out[i * 10 + 9] = W.win_16384;
    }
}

// 取走段统计（count,sum,best,max,1024,2048,4096,8192,16384）并清零；返回段内局数
__declspec(dllexport) int64_t ai_train_seg_take(int wid, int64_t* out9) {
    if (wid < 0 || wid >= (int)g_workers.size()) return 0;
    Worker& W = *g_workers[wid];
    std::lock_guard<std::mutex> lk(W.mtx);
    out9[0] = W.seg_n; out9[1] = W.seg_sum; out9[2] = W.seg_best; out9[3] = W.seg_max;
    out9[4] = W.seg_1024; out9[5] = W.seg_2048; out9[6] = W.seg_4096; out9[7] = W.seg_8192;
    out9[8] = W.seg_16384;
    int64_t n = W.seg_n;
    W.seg_n = W.seg_sum = W.seg_best = W.seg_max = 0;
    W.seg_1024 = W.seg_2048 = W.seg_4096 = W.seg_8192 = W.seg_16384 = 0;
    return n;
}

} // extern "C"
