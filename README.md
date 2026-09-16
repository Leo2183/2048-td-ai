# 2048-TD-AI

一个会自我进化着玩 2048 的本地项目：从零开始的强化学习 AI，靠"每一步合并得分"作为唯一奖励信号，自我对弈数百万局，练到能稳定合出 16384。

![平台](https://img.shields.io/badge/%E5%B9%B3%E5%8F%B0-Windows-blue) ![语言](https://img.shields.io/badge/%E8%AF%AD%E8%A8%80-Python%20%2B%20C%2B%2B-green) ![许可](https://img.shields.io/badge/License-MIT-yellow)

## 战力一览

| 指标 | 数值（训练口径） |
|---|---|
| 累计训练 | 287 万局自我对弈 |
| 平均分 | ~172,000（从随机水平的 ~900 起步） |
| 8192 达成率 | 87% |
| 16384 达成率 | 26%+（实战深搜口径更高） |

打开游戏窗口点「AI 演示」，可以实时观看它下棋——支持慢速/快速/极速三档回放速度。

## 项目组成

| 文件 | 说明 |
|---|---|
| `game2048.py` | tkinter 游戏本体：动画 UI、人玩、AI 演示模式（S 键切换速度档） |
| `ai2048.py` | 算法核心：N-tuple 网络、afterstate TD(0) 学习、自适应 expectimax 搜索 |
| `engine.cpp` | C++ 引擎：位棋盘、8 线程 Hogwild 训练、深搜（约 1800 局/秒） |
| `build_engine.py` | MSVC 一键构建 `engine.dll` |
| `train2048.py` | 命令行训练器 |
| `train_gui.py` | 训练器 GUI：实时统计、EMA 降噪曲线、CPU 占用控制、训练时阻止休眠 |
| `启动2048.vbs` / `训练AI.bat` | 双击即用的启动器 |

## 快速开始

### 环境依赖

- **必需**：Windows、Python 3.10+、numpy
- **建议**：numba（约 10 倍训练加速）
- **可选**：MSVC Build Tools（C++ 引擎再快约 14 倍；缺失时自动回退，功能不变）

### 三步跑起来

```bash
git clone https://github.com/Leo2183/2048-td-ai
cd 2048-td-ai
python build_engine.py     # 构建 C++ 引擎（无 MSVC 时跳过，自动用 numba）
```

- **看 AI 下棋**：双击 `启动2048.vbs` → 点「AI 演示」
- **自己训练**：双击 `训练AI.bat`（或 `python train_gui.py`）→ 点「开始训练」，模型自动从零开始，随时暂停/停止/关窗，自动保存续训

### 训练调参指南（实踩经验）

学习率 α 是唯一重要旋钮，按训练量退火：

| 累计局数 | 建议 α | 说明 |
|---|---|---|
| 起步 | 0.005 | 快速吸收主线信号 |
| ~20 万局 | 0.001 | 容量大的网络步长过大会震荡掉档（本项目实测踩过） |
| 曲线横盘后 | 0.0005 | 最后一段精修 |

其余交给挂机：C++ 引擎约 1800 局/秒，一晚数百万局。GUI 的实时曲线（平均分红线、8192 率金线、16384 率青线，均带 EMA 降噪）就是进度表。

## 技术说明

**学习**（Szubert & Jaśkowski 2014 的经典路线，本项目的 numpy/C++ 复现）：

- N-tuple 特征网络：8 个 6 格棋盘形状 × 旋转镜像对称 = 64 个查表打分器，估值 = 打分之和。表共 512MB float32
- Afterstate TD(0)：每走一步，用"预期 vs 实际得分差"修正上一步局面的估值；奖励就是合并得分，终局价值为 0
- 8 线程 Hogwild：各线程独立对弈、无锁写同一张权重表

**实战决策**：

- 自适应深度 expectimax（开局 1 层 → 终盘 4 层），空格多时剪掉 10% 的 4 块分支
- **贪心叶对齐**：价值函数定义在 afterstate 上，直接评估出块后盘面会产生分布偏移（实测反而比不搜索更差），故叶节点先贪心展开一步再评估
- 置换表缓存子树结果（纯记忆化，棋力零影响，残局提速 2.5×）

**工程**：

- 三级引擎回退：C++（ctypes，零第三方依赖）→ numba → 纯 numpy
- 训练统计三级流水：0.4s 实时卡片 → 2s 曲线采样 → 每 250 局存档
- 训练时 `SetThreadExecutionState` 阻止系统休眠

## 参考

- Szubert & Jaśkowski, *Temporal Difference Learning of N-Tuple Networks for the Game 2048* (CEC 2014)
- Jaśkowski et al., *Mastering 2048 with Shift-Invariant N-Tuples* (GECCO 2015)
- Yeh et al., *Multi-Stage Temporal Difference Learning for 2048-like Games* (2016)
- [TDL2048](https://github.com/moporgic/TDL2048)（同类最强开源实现，本项目的对照标杆）

## License

[MIT](LICENSE) © Leo2183
