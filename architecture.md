# Beer Game RL 重构架构文档

## 1. 整体技术结构

```
coursebeergame/
├── envs/
│   └── beer_game_env.py          # 唯一的环境实现（向后兼容 + Lead Time + 需求分布扩展）
├── algos/
│   ├── base_agent.py             # 统一智能体接口
│   ├── builder.py                # 工厂函数，按算法名动态构造智能体
│   ├── dqn/
│   │   ├── agent.py              # DQN 智能体
│   │   └── network.py            # Q 网络（DQN/DDQN 复用）
│   ├── ddqn/
│   │   └── agent.py              # Double DQN 智能体
│   ├── ppo_ippo/
│   │   ├── agent.py              # PPO 单智能体 + IPPO 多智能体包装
│   │   └── network.py            # Actor-Critic 网络
│   └── mappo/
│       ├── agent.py              # MAPPO（CTDE：局部 Actor + 集中式 Critic）
│       └── network.py            # Actor / Centralized Critic 网络
├── utils/
│   ├── seed.py                   # 全局随机种子
│   ├── logger.py                 # CSV 训练日志
│   ├── replay_buffer.py          # DQN/DDQN 经验回放
│   ├── rollout_buffer.py         # PPO/IPPO/MAPPO 轨迹缓冲
│   ├── plotting.py               # 统一画图（学习曲线、测试面板、牛鞭效应、敏感性热力图）
│   ├── heuristic.py              # 启发式策略（Order-Up-To / Random）
│   └── config.py                 # YAML 配置加载
├── configs/
│   ├── default.yaml              # 环境默认参数
│   └── algo/
│       ├── dqn.yaml              # DQN 超参
│       ├── ddqn.yaml             # DDQN 超参
│       ├── ppo.yaml              # PPO 超参
│       ├── ippo.yaml             # IPPO 超参
│       └── mappo.yaml            # MAPPO 超参
├── main_single.py                # 单智能体入口（DQN/DDQN/PPO）
├── main_multi.py                 # 多智能体入口（IPPO/MAPPO）
├── compare.py                    # 批量对比实验脚本
├── requirements.txt              # 依赖
└── architecture.md               # 本文档
```

### 1.1 环境（`envs/beer_game_env.py`）

- 将原先散落在三个脚本中的 `Env` 类合并为 `BeerGameEnv`。
- 观测空间：
  - `lead_time == 0`：3 维 `[订单, 满足需求, 库存]`，与原代码完全一致。
  - `lead_time > 0`：4 维，额外拼接 `pipeline_inventory`（在途库存总量）。
- 需求分布（`demand_type`）：
  - `poisson`：默认泊松需求。
  - `seasonal`：带正弦波动的泊松需求。
  - `shock`：以小概率叠加突发大订单。
- 提前期（`lead_time`）：每个企业维护长度为 `L` 的 FIFO 到货队列；本步订单 `L` 步后到货。

### 1.2 算法模块（`algos/`）

- `BaseAgent` 规定统一接口：`act / step / update / save / load`。
- `builder.build_agent(algo_name, ...)` 根据字符串动态导入对应算法，新增算法只需新建目录并在 `builder.py` 注册。
- DQN/DDQN：离散动作 `0 ~ max_order`，使用共享 `QNetwork` 与 `ReplayBuffer`。
- PPO：Actor-Critic 共享网络、`RolloutBuffer`、GAE、PPO-clip、奖励归一化。
- IPPO：`N` 个独立 PPO 智能体，互不共享参数与经验。
- MAPPO：每个企业独立局部 Actor，Critic 输入全局状态（所有企业观测拼接），实现 CTDE。

### 1.3 工具模块（`utils/`）

- `plotting.py` 提供 5 类图：单次训练曲线、多算法/多种子学习曲线、测试指标面板、订购行为对比、牛鞭效应柱状图、环境敏感性热力图。
- `logger.py` 以 CSV 记录每回合奖励，便于对比脚本聚合。
- `seed.py` 固定 Python / NumPy / PyTorch 随机种子。

## 2. 运行实验方式

### 2.1 安装依赖

```bash
pip install -r requirements.txt
```

依赖：`numpy`, `torch`, `matplotlib`, `seaborn`, `pandas`, `pyyaml`。

### 2.2 单智能体实验

控制指定 `firm_id`，其余企业使用启发式策略（默认 `order_up_to`）。

```bash
# DQN / DDQN / PPO，标准环境
python main_single.py --alg=dqn   --firm_id=1 --episodes=2000 --lead_time=0 --demand=poisson --seed=0
python main_single.py --alg=ddqn  --firm_id=1 --episodes=2000 --lead_time=0 --demand=poisson --seed=0
python main_single.py --alg=ppo   --firm_id=1 --episodes=2000 --lead_time=0 --demand=poisson --seed=0

# 扩展环境：提前期 + 季节性需求
python main_single.py --alg=ppo   --firm_id=1 --episodes=2000 --lead_time=2 --demand=seasonal --seed=1

# 其余企业使用随机策略
python main_single.py --alg=dqn   --firm_id=1 --lead_time=0 --demand=poisson --heuristic=random
```

### 2.3 多智能体实验

`num_agents` 个企业全部由 RL 控制。

```bash
python main_multi.py --alg=ippo   --num_agents=3 --episodes=3000 --lead_time=2 --demand=seasonal --seed=0
python main_multi.py --alg=mappo  --num_agents=3 --episodes=3000 --lead_time=2 --demand=shock    --seed=0
```

### 2.4 批量对比实验

`compare.py` 自动跑网格实验并汇总画图。

```bash
# 跑 DQN/DDQN/PPO，3 个种子，标准环境
python compare.py --algs=dqn,ddqn,ppo --firm_id=1 --seeds=0,1,2 --lead_times=0 --demands=poisson

# 跑环境敏感性网格：poisson/seasonal/shock × lead_time 0/2
python compare.py --algs=ppo,mappo --firm_id=1 --seeds=0,1,2 --lead_times=0,2 --demands=poisson,seasonal,shock

# 仅基于已生成的 CSV 画图（不重新训练）
python compare.py --algs=dqn,ddqn,ppo --firm_id=1 --seeds=0,1,2 --lead_times=0 --demands=poisson --skip_run
```

## 3. 数据查看方式

### 3.1 训练日志（CSV）

每次运行会在 `results/` 下生成 CSV，例如：

```
results/dqn_single_firm1_lt0_demandpoisson_seed0.csv
results/ippo_n3_lt2_demandseasonal_seed0.csv
```

直接用 Excel / pandas 读取：

```python
import pandas as pd
df = pd.read_csv("results/dqn_single_firm1_lt0_demandpoisson_seed0.csv")
print(df.head())
```

### 3.2 训练曲线与测试面板（PNG）

图片保存在 `figures/`：

- 单次实验：`figures/{alg}_single_firm{firm_id}_lt{lt}_demand{demand}_seed{seed}_train.png`
- 测试面板：`figures/{alg}_single_firm{firm_id}_lt{lt}_demand{demand}_seed{seed}_test.png`
- 对比学习曲线：`figures/compare_lt{lt}_demand{demand}_firm{firm_id}_learning.png`
- 敏感性热力图：`figures/compare_{alg}_firm{firm_id}_sensitivity.png`

### 3.3 牛鞭效应指标

`main_multi.py` 测试阶段会自动打印各企业 `Var(order) / Var(demand)`，例如：

```
牛鞭效应指标 Var(order)/Var(demand): ['1.000', '2.350', '5.120']
```

数值越大表示越上游企业订单波动放大越严重，可用于对比 IPPO / MAPPO / 启发式策略的协调能力。

### 3.4 模型权重

训练好的模型保存在 `models/`：

```
models/dqn_firm_1_final.pth
models/ippo_n3_final_agent_0.pth
models/mappo_n3_final_agent_0.pth
```

可通过对应 `Agent.load(path)` 加载并继续训练或测试。

## 4. 扩展提示

- **新增算法**：在 `algos/` 下新建目录，实现 `BaseAgent` 接口，然后在 `algos/builder.py` 注册。
- **新增需求类型**：在 `BeerGameEnv._current_lambda()` 与 `_generate_demand()` 中添加分支即可。
- **新增对比维度**：修改 `compare.py` 的解析与循环，或直接用 CSV + `utils/plotting.py` 自定义画图。

## 5. 已知限制

- 当前沙箱环境因网络/权限原因无法完整安装 `torch`，因此未执行含 PyTorch 的端到端训练测试；环境动力学与画图模块已独立验证通过。
- 建议在本地或 GPU/CPU 环境安装 `requirements.txt` 后运行完整实验。
