# 基于深度强化学习的啤酒游戏供应链

本项目完整、可复现地实现了啤酒游戏(Beer Game)供应链基准测试，
采用五种深度强化学习算法(DQN、Double DQN、PPO、IPPO、MAPPO)，
并将其扩展至考虑提前期、非泊松需求分布以及多智能体 CTDE 设置的情形。

本项目为北京大学课程 *多智能体基础* 的配套实现。

## 什么是啤酒游戏？

啤酒游戏是一种广泛使用的游戏，由多名玩家组成的串行供应链网络组成：

1. 零售商：接收客户的订单并向批发商下订单；
2. 中间商：接收上游企业的订单并向零售商下订单。

```
                  订购                       订购                    订购
上游 ──交付──► 中间商 ──交付──► … ──交付──► 中间商 ──交付──► 零售商 ──► 满足用户需求
```

供应链中的企业根据观察到的订单和需求等信息，决策向上游订购的订单水平。
研究对象可以是链上企业（在作业代码中定义 3 个），可以选择一个企业进行
利润优化，将其余看作是环境的一部分。

由于每家企业都在局部上独立决策，客户需求的小幅波动会沿供应链向上游
传递时被不断放大——这就是著名的 *牛鞭效应* (bullwhip effect, Lee,
Padmanabhan & Whang, 1997; Sterman, 1989)。

本项目沿用初始代码中的三企业链结构，并将订货量视为深度强化学习
智能体选择的离散动作。

## 项目结构

```
coursebeergame/
├── envs/
│   └── beer_game_env.py          # BeerGameEnv：支持提前期与 3 种需求模式
├── algos/
│   ├── base_agent.py             # 所有智能体的抽象基类
│   ├── builder.py                # 智能体工厂 build_agent(name, ...)
│   ├── dqn/{agent,network}.py    # DQN 与共享 Q 网络
│   ├── ddqn/agent.py             # Double DQN
│   ├── ppo_ippo/
│   │   ├── agent.py              # PPO 与 IPPO 包装器 (N 个独立 PPO)
│   │   └── network.py            # Actor-Critic
│   └── mappo/
│       ├── agent.py              # MAPPO 单智能体与多智能体包装器
│       └── network.py            # 局部 Actor + 中心化 Critic (CTDE)
├── utils/
│   ├── config.py                 # YAML 配置加载器
│   ├── seed.py                   # 全局随机数种子
│   ├── logger.py                 # CSV 训练日志
│   ├── replay_buffer.py          # DQN/DDQN 经验回放
│   ├── rollout_buffer.py         # PPO/IPPO/MAPPO 采样存储
│   ├── plotting.py               # 学习曲线、牛鞭效应、敏感性图
│   └── heuristic.py              # order-up-to 策略、随机策略
├── configs/
│   ├── default.yaml              # 环境默认配置
│   └── algo/{dqn,ddqn,ppo,ippo,mappo}.yaml
├── main_single.py                # 单智能体入口：DQN / DDQN / PPO
├── main_multi.py                 # 多智能体入口：IPPO / MAPPO
├── compare.py                    # 网格实验驱动与图表生成
├── envs/ figures/ models/ results/   # 由脚本运行生成
├── report.tex, report_cn.tex     # 实验报告 LaTeX 源文件 (英文 / 中文)
├── report.pdf, report_cn.pdf     # 编译后的报告
├── report.bib                    # 参考文献
├── architecture.md               # 详细架构说明
├── 组队课题.pdf                  # 课程课题说明
└── requirements.txt
```

## 环境安装

```bash
# 在 Python 3.8+、PyTorch 1.10+ 下测试通过
pip install -r requirements.txt
```

`requirements.txt` 内容：

```
numpy
torch
matplotlib
seaborn
pandas
pyyaml
```

GPU 可选——本项目在 CPU 上也能顺利运行。

## 快速上手

训练单个配置：

```bash
# 单智能体：使用 DQN 控制企业 1，无提前期，泊松需求
python main_single.py --alg=dqn --firm_id=1 --episodes=2000 --lead_time=0 --demand=poisson --seed=0

# 单智能体：使用 PPO，提前期 2 步，季节性需求
python main_single.py --alg=ppo --firm_id=1 --episodes=2000 --lead_time=2 --demand=seasonal --seed=1

# 多智能体：使用 IPPO 控制全部 3 家企业
python main_multi.py --alg=ippo --num_agents=3 --episodes=3000 --lead_time=0 --demand=seasonal --seed=0

# 多智能体：使用 MAPPO，提前期 2 步
python main_multi.py --alg=mappo --num_agents=3 --episodes=3000 --lead_time=2 --demand=poisson --seed=1
```

输出文件：

- `results/{alg}_{config}_seed{N}.csv` — 每个 episode 的奖励。
- `results/{alg}_{config}_seed{N}_bullwhip.json` — 每家企业的订单/需求方差。
- `models/{alg}_*.pth` — 在 episode 500/1000/1500/2000 与最终 episode 保存的检查点。
- `figures/{alg}_..._train.png` 与 `_test.png` — 单次运行的训练曲线与测试仪表板。

运行完整网格实验 (5 个算法 × 2 个提前期 × 2 种需求模式 × 3 个种子 = 60 次运行)：

```bash
python compare.py --algs=dqn,ddqn,ppo,ippo,mappo --seeds=0,1,2 --lead_times=0,2 --demands=poisson,seasonal
```

生成的图表：

- `compare_single_lt{LT}_demand{DEM}_firm1_learning.png` — DQN/DDQN/PPO 学习曲线。
- `compare_multi_lt{LT}_demand{DEM}_n3_learning.png` — IPPO/MAPPO 学习曲线。
- `compare_{alg}_firm1_sensitivity.png` 与 `_n3_sensitivity.png` — 环境敏感性热力图。
- `compare_bullwhip_single_*.png` / `_multi_*.png` — 牛鞭效应柱状图。

仅基于已保存的 CSV 重新渲染图表：

```bash
python compare.py --algs=dqn,ddqn,ppo,ippo,mappo --seeds=0,1,2 --lead_times=0,2 --demands=poisson,seasonal --skip_run
```

## 算法一览

| 算法 | 类 | 骨干网络 | 备注 |
|-----------|-------|----------|-------|
| DQN       | `algos.dqn.agent.DQNAgent`        | 3-64-64-21 MLP | 目标网络、经验回放、软更新 |
| Double DQN | `algos.ddqn.agent.DDQNAgent`    | 共享 Q 网络 | 解耦动作选择与价值估计 |
| PPO       | `algos.ppo_ippo.agent.PPOAgent`  | Actor-Critic | GAE、截断替代目标、K=3 epoch、熵正则 |
| IPPO      | `algos.ppo_ippo.agent.IPPOAgent` | N × PPO | 每家企业独立运行 PPO，无通信 |
| MAPPO     | `algos.mappo.agent.MAPPO`        | 局部 Actor + 中心化 Critic | CTDE：Critic 看到所有企业观测的拼接 |

所有智能体均实现 `BaseAgent` 接口 (`act`、`step`、`update`、`save`、`load`)。
新增算法只需在 `algos/` 下创建新目录并在 `algos.builder` 中注册一行即可。

## 环境

`BeerGameEnv` (位于 `envs/beer_game_env.py`) 暴露以下接口：

- `num_firms` (默认 3) 家串联企业：零售商 (0)、批发商 (1)、制造商 (2)。
- 企业 0 的需求：`demand_type ∈ {poisson, seasonal, shock}`。
- 企业 `i ≥ 1` 的需求：等于企业 `i-1` 上一步的订货量。
- 提前期 `L`：每笔订单需 `L` 步到达。`L>0` 时，智能体观测从 3 维扩展为 4 维，
  增加一维标量 *管道库存* (在途订单之和)。
- 动作：取值范围 `[0, max_order]` 的整数订货量 (默认 20)。
- 奖励：`p[i] * 满足量 - p[i+1] * 订货量 - h * 库存 - c * 缺货损失`。

默认参数 (位于 `configs/default.yaml`)：

| 参数 | 值 |
|-----------|-------|
| `num_firms`              | 3 |
| `p` (销售价)             | [10, 9, 8] |
| `h` (持有成本)           | 0.5 |
| `c` (缺货惩罚)           | 2 |
| `initial_inventory`      | 100 |
| `poisson_lambda`         | 10 |
| `max_steps`              | 100 |
| `max_order`              | 20 |

## 超参数

每个算法在 `configs/algo/` 下有对应 YAML，默认值如下：

| 超参数 | DQN | DDQN | PPO | IPPO | MAPPO |
|---|---|---|---|---|---|
| learning rate / lr | 1e-3 | 1e-3 | 3e-4 | 3e-4 | 3e-4 |
| gamma              | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 |
| buffer size        | 10k | 10k | — | — | — |
| batch size         | 64 | 64 | — | — | — |
| soft update tau    | 1e-3 | 1e-3 | — | — | — |
| eps decay          | 0.995→0.01 | 0.995→0.01 | — | — | — |
| PPO clip epsilon   | — | — | 0.2 | 0.2 | 0.2 |
| K epochs           | — | — | 3 | 3 | 3 |
| GAE lambda         | — | — | 0.95 | 0.95 | 0.95 |
| entropy coef       | — | — | 0.05→0.001 | 0.05→0.001 | 0.05→0.001 |
| 每 N 个 episode 更新 | 4 | 4 | 5 | 5 | 5 |
| 训练 episode 数     | 2000 | 2000 | 2000 | 3000 | 3000 |

## 复现实验报告

PDF 实验报告由 `report.tex` / `report_cn.tex` 与 `report.bib` 生成。
中文版使用 `xelatex` 编译，英文版使用 `pdflatex` 编译：

```bash
# 英文
pdflatex report.tex && bibtex report && pdflatex report.tex && pdflatex report.tex

# 中文
xelatex report_cn.tex && bibtex report_cn && xelatex report_cn.tex && xelatex report_cn.tex
```

自定义样式 `icml_simple.sty` 与 `icml_cn.sty` 提供类 ICML 的双栏排版，
不依赖完整的 ICML 2022 模板。它们假定 LaTeX 环境中已安装 `ctex`、
`titlesec`、`caption`、`natbib` 和 `hyperref`。

## 结果概览

通过 60 次网格扫描实验，我们得到以下五条主要经验性结论：

1. **单智能体 DQN/DDQN/PPO 表现无显著差异** —— 当邻居企业采用
   确定性的 Order-Up-To 启发式时，三者都会收敛为常量订货策略
   (订单方差为 0)，最后 100 个 episode 的奖励约为 715。
2. **IPPO 较为脆弱**：在 3 个季节性需求种子中有 1 个会出现灾难性发散，
   奖励跌至约 -25,000，即使将训练延长至 5,000 个 episode 仍无法恢复。
   MAPPO 在同一网格上从未失败。
3. **提前期会放大牛鞭效应**：在 L=2、泊松需求下，批发商位置的 MAPPO
   在某一种子上达到 Var(订单)/Var(需求) = 9.97。
4. **可预测的需求是学得会的**：在季节性需求下，IPPO 实现了
   bw = [0.99, 0.76, 0.52] (上游订单方差 *小于* 需求方差——方差衰减)。
5. **提前期的惩罚与算法无关**：所有算法在从 L=0 变为 L=2 时，
   奖励均下降 1,100–1,700。

## 文件命名规范

- 单智能体 CSV：`results/{alg}_single_firm{firm}_lt{LT}_demand{DEM}_seed{N}.csv`
- 多智能体 CSV：`results/{alg}_n{N}_lt{LT}_demand{DEM}_seed{N}.csv`
- 牛鞭效应 JSON：上述前缀 + `_bullwhip.json`
- 模型检查点：`models/{alg}_firm{firm}_ep{N}.pth` (单智能体) 或
  `models/{alg}_n{N}_agent{i}.pth` (多智能体)
- 图表：`figures/{alg}_...` 为单次运行，`figures/compare_...` 为批量扫描

## 扩展指南

- 新增算法：创建 `algos/<name>/agent.py`，在 `algos.builder` 中注册一行，
  并在 `configs/algo/` 下添加 YAML。
- 新增需求模式：扩展 `BeerGameEnv._current_lambda` / `_generate_demand`
  以及 `configs/default.yaml`。
- 新增图表类型：在 `utils/plotting.py` 中实现，并接入 `compare.py`。

## 致谢

啤酒游戏动力学遵循标准形式 (Sterman, 1989; Lee 等, 1997)。
DQN、PPO 和 SAC 基线参考自原始论文
(Mnih 等, 2015; Schulman 等, 2017; Haarnoja 等, 2018)。

---

作者：匿名 (北京大学, REDACTED_ID)  
课程：多智能体基础 (Multi-Agent Fundamentals), 2026  
助教：REDACTED_TA (redacted@stu.pku.edu.cn)
