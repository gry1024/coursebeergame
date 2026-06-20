import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os


# ============================================================
# 环境类：多阶供应链库存管理仿真环境
# ============================================================
class Env:
    def __init__(self, num_firms, p, h, c, initial_inventory, poisson_lambda=10, max_steps=100):
        """
        初始化供应链管理仿真环境。

        :param num_firms: 企业数量
        :param p: 各企业的价格列表，p[i] 表示第 i 个企业售出产品的单价
        :param h: 库存持有成本（每单位库存每步的成本）
        :param c: 损失销售成本（每单位未满足需求的惩罚成本）
        :param initial_inventory: 每个企业的初始库存
        :param poisson_lambda: 最下游企业（firm 0）面临的外部需求服从泊松分布的均值
        :param max_steps: 每个 episode 的最大步数
        """
        self.num_firms = num_firms
        self.p = p                       # 企业的价格列表
        self.h = h                       # 库存持有成本
        self.c = c                       # 损失销售成本
        self.poisson_lambda = poisson_lambda  # 泊松分布的均值
        self.max_steps = max_steps       # 每个 episode 的最大步数
        self.initial_inventory = initial_inventory  # 初始库存

        # 初始化库存为列向量，形状 (num_firms, 1)
        self.inventory = np.full((num_firms, 1), initial_inventory)
        # 初始化订单量为零
        self.orders = np.zeros((num_firms, 1))
        # 初始化已满足的需求量为零
        self.satisfied_demand = np.zeros((num_firms, 1))
        # 记录当前步数
        self.current_step = 0
        # 标记 episode 是否结束
        self.done = False

    def reset(self):
        """
        重置环境到初始状态，开始新的 episode。

        :return: 初始观察值，形状 (num_firms, 3)
        """
        self.inventory = np.full((self.num_firms, 1), self.initial_inventory)
        self.orders = np.zeros((self.num_firms, 1))
        self.satisfied_demand = np.zeros((self.num_firms, 1))
        self.current_step = 0
        self.done = False
        return self._get_observation()

    def _get_observation(self):
        """
        获取每个企业的局部观察信息。
        每个企业的状态是独立的，包含 [订单量, 满足的需求量, 库存量]。

        :return: 观察矩阵，形状 (num_firms, 3)
        """
        return np.concatenate((self.orders, self.satisfied_demand, self.inventory), axis=1)

    def _generate_demand(self):
        """
        根据规则生成每个企业在本时间步的需求。

        - 最下游企业（firm 0）的需求服从泊松分布 Poisson(poisson_lambda)。
        - 其他上游企业的需求等于其直接下游企业的订单量，即 d_{i,t} = q_{i-1,t}。

        :return: 需求向量，形状 (num_firms, 1)
        """
        demand = np.zeros((self.num_firms, 1))
        for i in range(self.num_firms):
            if i == 0:
                # 最下游企业的外部需求服从泊松分布
                demand[i] = np.random.poisson(self.poisson_lambda)
            else:
                # 上游企业的需求等于下游企业的订单量
                demand[i] = self.orders[i - 1]
        return demand

    def step(self, actions):
        """
        执行一个时间步的仿真，根据给定的行动（每个企业的订单量）更新环境状态。

        :param actions: 每个企业的订单量，形状 (num_firms, 1)
        :return: next_state, rewards, done
                 next_state: 下一时刻观察，形状 (num_firms, 3)
                 rewards: 每个企业的即时奖励，形状 (num_firms, 1)
                 done: episode 是否结束
        """
        # 更新本步各企业的订单量
        self.orders = actions

        # 生成各企业的需求
        self.demand = self._generate_demand()

        # 计算每个企业的满足需求量（受限于当前库存）
        for i in range(self.num_firms):
            self.satisfied_demand[i] = min(self.demand[i], self.inventory[i])

        # 更新库存：新库存 = 原库存 + 本步订货量 - 本步满足的需求量
        for i in range(self.num_firms):
            self.inventory[i] = self.inventory[i] + self.orders[i] - self.satisfied_demand[i]

        # 计算每个企业的奖励
        # 奖励 = 销售收入 - 采购成本 - 库存持有成本 - 缺货损失成本
        # 销售收入：p[i] * satisfied_demand[i]
        # 采购成本：p[i+1] * orders[i]（向上游进货的成本，最上游企业无上游采购成本）
        # 库存持有成本：h * inventory[i]
        # 缺货损失成本：c * (demand[i] - satisfied_demand[i])
        rewards = np.zeros((self.num_firms, 1))
        loss_sales = np.zeros((self.num_firms, 1))

        for i in range(self.num_firms):
            revenue = self.p[i] * self.satisfied_demand[i]
            purchase_cost = (self.p[i + 1] if i + 1 < self.num_firms else 0) * self.orders[i]
            holding_cost = self.h * self.inventory[i]
            rewards[i] += revenue - purchase_cost - holding_cost

            # 损失销售费用（缺货惩罚）
            if self.satisfied_demand[i] < self.demand[i]:
                loss_sales[i] = (self.demand[i] - self.satisfied_demand[i]) * self.c

        # 总奖励扣除损失销售成本
        rewards -= loss_sales

        # 增加步数
        self.current_step += 1

        # 判断是否达到最大步数，结束 episode
        if self.current_step >= self.max_steps:
            self.done = True

        return self._get_observation(), rewards, self.done


# 供应链基准策略：Order-Up-To（订货至目标水平）策略
def order_up_to_policy(state, firm_id, target_inventory=120, max_order=20):
    """
    Order-Up-To 策略：基于当前库存，将库存补充到目标水平。
    注意：这不是真正的 (s, S) 策略，因为它没有再订货点 s，
          只要库存低于目标水平就会订货。

    :param state: 当前环境观察，形状 (num_firms, 3)
    :param firm_id: 企业 ID
    :param target_inventory: 目标库存水平 S
    :param max_order: 最大订单量限制
    :return: 该企业的订单量（整数）
    """
    current_inventory = state[firm_id][2]  # 库存是第 3 列（索引 2）
    order = max(0, target_inventory - current_inventory)
    order = min(order, max_order)          # 限制最大订单量
    return max(0, int(order))


# 真正的 (s, S) 策略
def ss_policy(state, firm_id, reorder_point=80, order_up_to_level=120, max_order=20):
    """
    真正的 (s, S) 策略：
    - 当库存低于或等于再订货点 s 时，订货将库存补充至 S；
    - 否则不订货。

    :param state: 当前环境观察，形状 (num_firms, 3)
    :param firm_id: 企业 ID
    :param reorder_point: 再订货点 s
    :param order_up_to_level: 目标库存水平 S
    :param max_order: 最大订单量限制
    :return: 该企业的订单量（整数）
    """
    current_inventory = state[firm_id][2]  # 库存是第 3 列（索引 2）
    if current_inventory <= reorder_point:
        order = max(0, order_up_to_level - current_inventory)
        order = min(order, max_order)
        return max(0, int(order))
    else:
        return 0


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# PPO 神经网络：同时包含 Actor（策略）和 Critic（价值）
class PPONetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        """
        初始化 PPO 网络。
        :param state_size: 状态空间维度（每个企业的局部状态维度为 3）
        :param action_size: 动作空间维度（对应可选的订单量数量）
        :param hidden_size: 隐藏层维度
        """
        super(PPONetwork, self).__init__()
        # 共享的特征提取层
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # Actor 头：输出每个动作的概率分布
        self.actor = nn.Linear(hidden_size, action_size)
        # Critic 头：输出状态价值 V(s)
        self.critic = nn.Linear(hidden_size, 1)
        
        # 对 Actor 最后一层做较小的初始化，避免初始策略过于尖锐
        self.actor.weight.data.mul_(0.01)
        self.actor.bias.data.mul_(0.0)

    def forward(self, state):
        """
        前向传播。
        :param state: 输入状态张量
        :return: action_probs（动作概率分布）, state_value（状态价值）
        """
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action_probs = torch.softmax(self.actor(x), dim=-1)
        state_value = self.critic(x)
        return action_probs, state_value

    def evaluate(self, states, actions):
        """
        批量计算给定状态下，某动作的对数概率、分布熵和状态价值。
        该函数用于 PPO 的损失计算。

        :param states: 状态张量，形状 (batch_size, state_size)
        :param actions: 动作索引张量，形状 (batch_size,)
        :return: log_probs（对数概率）, state_values（状态价值）, entropy（分布熵）
        """
        action_probs, state_values = self.forward(states)
        # 使用 Categorical 分布表示离散动作空间
        dist = torch.distributions.Categorical(action_probs)
        # 计算选中动作的对数概率
        log_probs = dist.log_prob(actions)
        # 计算策略熵，用于鼓励探索并防止策略过早坍塌
        entropy = dist.entropy()
        return log_probs, state_values.squeeze(-1), entropy


# PPO 经验回放缓冲区
class PPOMemory:
    def __init__(self):
        """
        初始化 PPO 的经验缓冲区，用于存储一个回合的数据。
        """
        self.states = []      # 状态
        self.actions = []     # 动作索引
        self.log_probs = []   # 旧策略下动作的对数概率
        self.rewards = []     # 即时奖励
        self.dones = []       # 是否结束标记
        self.values = []      # 旧策略下的状态价值估计

    def push(self, state, action, log_prob, reward, done, value):
        """
        存入一条经验。
        """
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def clear(self):
        """
        清空缓冲区。
        """
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

    def get(self):
        """
        获取缓冲区中所有数据，并转换为 PyTorch 张量。

        :return: states, actions, log_probs, rewards, dones, values
        """
        states = torch.FloatTensor(np.array(self.states)).to(device)
        actions = torch.LongTensor(np.array(self.actions)).to(device)
        log_probs = torch.FloatTensor(np.array(self.log_probs)).to(device)
        rewards = torch.FloatTensor(np.array(self.rewards)).to(device)
        dones = torch.FloatTensor(np.array(self.dones)).to(device)
        values = torch.FloatTensor(np.array(self.values)).to(device)
        return states, actions, log_probs, rewards, dones, values

    def __len__(self):
        return len(self.states)


# 奖励归一化器（running statistics）
class RewardNormalizer:
    """
    使用滑动平均对奖励进行归一化。
    将奖励缩放到接近 N(0, 1) 的分布，有助于：
    - 价值函数更快地学习
    - 降低策略梯度的方差
    - 防止策略因奖励尺度问题而坍塌
    """
    def __init__(self, clip=10.0):
        """
        :param clip: 归一化后奖励的裁剪范围，防止异常值
        """
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4
        self.clip = clip

    def update(self, reward):
        """
        使用 Welford 在线算法更新奖励的均值和方差。
        """
        self.count += 1
        delta = reward - self.mean
        self.mean += delta / self.count
        delta2 = reward - self.mean
        self.var += delta * delta2

    def normalize(self, reward, update=True):
        """
        对单个奖励进行归一化并裁剪。

        :param reward: 原始奖励
        :param update: 是否更新 running statistics
        :return: 归一化后的奖励
        """
        if update:
            self.update(reward)
        std = np.sqrt(self.var / self.count) + 1e-8
        normalized = (reward - self.mean) / std
        return np.clip(normalized, -self.clip, self.clip)


# PPO 智能体
class PPOAgent:
    """
    PPO（Proximal Policy Optimization）智能体类。
    算法特性：
    - 重要性采样比裁剪（Clipped Surrogate Objective）
    - 广义优势估计（GAE）
    - 共享参数的 Actor-Critic 网络结构
    """
    def __init__(self, state_size, action_size, firm_id, max_order=20,
                 gamma=0.99, lr=3e-4, eps_clip=0.2, K_epochs=3,
                 gae_lambda=0.95, entropy_coef=0.05, min_entropy_coef=0.001,
                 entropy_decay=0.995, value_coef=0.5):
        """
        初始化 PPO 智能体。

        :param state_size: 状态空间维度（每个企业局部状态为 3）
        :param action_size: 动作空间维度（可选订单量数量）
        :param firm_id: 当前智能体对应的企业 ID
        :param max_order: 最大订单量
        :param gamma: 奖励折扣因子
        :param lr: 网络学习率
        :param eps_clip: PPO 裁剪阈值，限制策略更新幅度
        :param K_epochs: 每个批次数据重复训练的轮数，建议 3 左右，过大易导致策略坍塌
        :param gae_lambda: GAE 中的 lambda 参数，平衡偏差与方差
        :param entropy_coef: 初始熵奖励系数，鼓励探索，防止策略过早坍塌
        :param min_entropy_coef: 熵奖励系数下限
        :param entropy_decay: 每次更新后熵奖励系数的衰减率
        :param value_coef: 价值损失系数
        """
        self.state_size = state_size
        self.action_size = action_size
        self.firm_id = firm_id
        self.max_order = max_order
        self.gamma = gamma
        self.lr = lr
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.min_entropy_coef = min_entropy_coef
        self.entropy_decay = entropy_decay
        self.value_coef = value_coef

        # 创建策略网络（当前策略，用于更新）
        self.policy_net = PPONetwork(state_size, action_size).to(device)
        # 创建旧策略网络（用于计算旧策略下的 log_prob，不直接更新）
        self.old_policy_net = PPONetwork(state_size, action_size).to(device)
        self.old_policy_net.load_state_dict(self.policy_net.state_dict())

        # Adam 优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

        # 经验缓冲区
        self.memory = PPOMemory()
        
        # 奖励归一化器，稳定价值函数学习并降低梯度方差
        self.reward_normalizer = RewardNormalizer()

    def decay_entropy(self):
        """
        每次策略更新后衰减熵奖励系数，让策略逐渐从探索转向利用。
        """
        self.entropy_coef = max(self.min_entropy_coef, self.entropy_coef * self.entropy_decay)

    def select_action(self, state, explore=True):
        """
        根据当前策略选择动作。

        :param state: 当前企业的局部状态
        :param explore: 是否进行探索。
                        True 时按概率分布采样动作（训练时使用）；
                        False 时直接选择概率最大的动作（测试时使用）。
        :return: action（实际订单量，整数）,
                 action_index（网络输出的动作索引，整数）,
                 log_prob（该动作的对数概率，标量张量）,
                 value（当前状态价值估计，标量张量）
        """
        state_tensor = torch.FloatTensor(state).to(device)
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)

        with torch.no_grad():
            action_probs, state_value = self.old_policy_net(state_tensor)
            dist = torch.distributions.Categorical(action_probs)

            if explore:
                # 训练时：按策略分布采样，天然带有探索性
                action_index = dist.sample()
            else:
                # 测试时：贪婪选择概率最大的动作
                action_index = torch.argmax(action_probs, dim=-1)

            log_prob = dist.log_prob(action_index)

        # 将网络输出的动作索引映射为实际订单量
        # 动作索引 0~max_order 直接对应订单量 0~max_order
        action = action_index.item()
        action = min(action, self.max_order)

        return action, action_index.item(), log_prob.item(), state_value.item()

    def store_transition(self, state, action_index, log_prob, reward, done, value):
        """
        存储一步转移数据到缓冲区。
        奖励会先经过 running normalization，以稳定训练。
        """
        normalized_reward = self.reward_normalizer.normalize(reward, update=True)
        self.memory.push(state, action_index, log_prob, normalized_reward, done, value)

    def compute_gae(self, rewards, values, dones, next_value):
        """
        使用 GAE（Generalized Advantage Estimation）计算优势函数和回报。

        :param rewards: 奖励序列，形状 (T,)
        :param values: 旧策略下的状态价值序列，形状 (T,)
        :param dones: 结束标记序列，形状 (T,)
        :param next_value: 最后状态的 bootstrapped 价值估计
        :return: returns（回报）, advantages（优势）
        """
        # 将 numpy 数组转换为 CUDA/CPU 张量，便于后续张量运算
        values = torch.FloatTensor(values).to(device)

        advantages = []
        gae = 0

        # 从后向前计算 GAE
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value_t = next_value
            else:
                next_value_t = values[t + 1].item()

            # 计算 TD 残差
            delta = rewards[t] + self.gamma * next_value_t * (1 - dones[t]) - values[t].item()
            # 递归计算 GAE
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        advantages = torch.FloatTensor(advantages).to(device)
        returns = advantages + values  # 回报 = 优势 + 价值
        return returns, advantages

    def update(self, next_state, done):
        """
        使用收集到的 rollout 数据更新 PPO 策略网络。

        :param next_state: rollout 结束后的下一状态，用于 bootstrapping
        :param done: 是否到达终止状态
        """
        if len(self.memory) == 0:
            return

        # 取出缓冲区数据
        states, actions, old_log_probs, rewards, dones, values = self.memory.get()

        # 计算最后状态的 bootstrapped 价值
        with torch.no_grad():
            next_state_tensor = torch.FloatTensor(next_state).to(device)
            if next_state_tensor.dim() == 1:
                next_state_tensor = next_state_tensor.unsqueeze(0)
            _, next_value_tensor = self.old_policy_net(next_state_tensor)
            next_value = next_value_tensor.item()

        # 如果已经到达终止状态，则 bootstrapped 价值为 0
        if done:
            next_value = 0.0

        # 使用 GAE 计算回报和优势
        returns, advantages = self.compute_gae(rewards.cpu().numpy(),
                                               values.cpu().numpy(),
                                               dones.cpu().numpy(),
                                               next_value)

        # 对优势进行归一化，有助于训练稳定性
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 对回报也做批归一化，进一步稳定 Critic 学习
        returns_normalized = (returns - returns.mean()) / (returns.std() + 1e-8)

        # PPO 核心：多次使用同一批数据更新策略
        for _ in range(self.K_epochs):
            # 在当前策略下重新计算动作的对数概率、状态价值和策略熵
            log_probs, state_values, entropy = self.policy_net.evaluate(states, actions)

            # 计算新旧策略的概率比
            ratios = torch.exp(log_probs - old_log_probs)

            # 裁剪后的替代目标
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic 损失：均方误差（使用归一化后的回报）
            critic_loss = nn.MSELoss()(state_values, returns_normalized)

            # 总损失：策略损失 + 价值损失 + 熵奖励（取负号因为我们要最大化熵）
            loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy.mean()

            # 反向传播并更新网络参数
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # 更新完成后，将当前策略复制到旧策略网络
        self.old_policy_net.load_state_dict(self.policy_net.state_dict())

        # 清空缓冲区
        self.memory.clear()

    def save(self, filepath):
        """
        保存当前策略网络参数。

        :param filepath: 模型保存路径
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.policy_net.state_dict(), filepath)
        print(f"模型已保存到: {filepath}")

    def load(self, filepath):
        """
        加载策略网络参数。

        :param filepath: 模型加载路径
        """
        self.policy_net.load_state_dict(torch.load(filepath, map_location=device))
        self.old_policy_net.load_state_dict(self.policy_net.state_dict())
        print(f"模型已从 {filepath} 加载")


# ============================================================
# PPO 训练函数
# ============================================================
def train_ppo(env, agent, num_episodes=2000, max_t=100, update_every=5):
    """
    训练 PPO 智能体。

    :param env: 供应链环境
    :param agent: PPOAgent 智能体
    :param num_episodes: 训练 episode 数量
    :param max_t: 每个 episode 的最大步数
    :param update_every: 每隔多少个 episode 更新一次策略。
                         例如 update_every=5 表示每收集 5 个 episode 的数据后更新一次。
                         标准 PPO 通常在收集足够多步数后更新，此处按 episode 聚合。
    :return: 所有 episode 的总奖励列表
    """
    scores = []  # 记录每个 episode 的总奖励
    episodes_since_update = 0  # 距离上次更新已经过的 episode 数
    last_state = None          # 最近一次 rollout 的结束状态
    last_done = False          # 最近一次 rollout 是否结束

    for i_episode in range(1, num_episodes + 1):
        state = env.reset()
        score = 0

        for t in range(max_t):
            # 为每个企业生成动作
            actions = np.zeros((env.num_firms, 1))
            action_index = None
            log_prob = None
            value = None

            for firm_id in range(env.num_firms):
                if firm_id == agent.firm_id:
                    # 当前智能体使用 PPO 策略选择动作（训练时开启探索）
                    firm_state = state[firm_id].reshape(1, -1)
                    action, action_index, log_prob, value = agent.select_action(firm_state, explore=True)
                    actions[firm_id] = action
                else:
                    # 其他企业使用 Order-Up-To 基准策略
                    actions[firm_id] = order_up_to_policy(state, firm_id)

            # 执行环境步进
            next_state, rewards, done = env.step(actions)

            # 获取当前智能体的奖励
            reward = rewards[agent.firm_id][0]
            score += reward

            # 存储转移数据
            agent.store_transition(
                state=state[agent.firm_id],
                action_index=action_index,
                log_prob=log_prob,
                reward=reward,
                done=done,
                value=value
            )

            # 状态更新
            state = next_state

            if done:
                break

        # 记录本 episode 的结束状态和是否终止
        last_state = state[agent.firm_id]
        last_done = done
        episodes_since_update += 1
        scores.append(score)

        # 当收集到 update_every 个 episode 后，执行一次策略更新
        if episodes_since_update >= update_every:
            agent.update(last_state, last_done)
            agent.decay_entropy()  # 更新后衰减熵奖励系数
            episodes_since_update = 0

        # 输出训练进度
        if i_episode % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print(f'Episode {i_episode}/{num_episodes} | Average Score: {avg_score:.2f}')

        # 每隔一定 episode 保存模型
        if i_episode % 500 == 0:
            agent.save(f'models/ppo_agent_firm_{agent.firm_id}_episode_{i_episode}.pth')

    # 训练结束时，如果还有未用于更新的经验，则再更新一次
    if episodes_since_update > 0:
        agent.update(last_state, last_done)
        agent.decay_entropy()

    # 训练结束后保存最终模型
    agent.save(f'models/ppo_agent_firm_{agent.firm_id}_final.pth')

    return scores


# ============================================================
# PPO 测试函数
# ============================================================
def test_agent(env, agent, num_episodes=10):
    """
    测试训练好的 PPO 智能体。

    :param env: 供应链环境
    :param agent: 训练好的 PPOAgent 智能体
    :param num_episodes: 测试 episode 数量
    :return: scores, inventory_history, orders_history, demand_history, satisfied_demand_history
    """
    scores = []
    inventory_history = []
    orders_history = []
    demand_history = []
    satisfied_demand_history = []

    for i_episode in range(1, num_episodes + 1):
        state = env.reset()
        score = 0
        episode_inventory = []
        episode_orders = []
        episode_demand = []
        episode_satisfied_demand = []

        for t in range(env.max_steps):
            # 为每个企业生成动作
            actions = np.zeros((env.num_firms, 1))
            for firm_id in range(env.num_firms):
                if firm_id == agent.firm_id:
                    # 当前智能体使用训练好的策略，测试时不探索（贪婪选择）
                    firm_state = state[firm_id].reshape(1, -1)
                    action, _, _, _ = agent.select_action(firm_state, explore=False)
                    actions[firm_id] = action
                else:
                    # 其他企业使用 Order-Up-To 基准策略
                    actions[firm_id] = order_up_to_policy(state, firm_id)

            # 执行环境步进
            next_state, rewards, done = env.step(actions)

            # 记录关键指标
            episode_inventory.append(env.inventory[agent.firm_id][0])
            episode_orders.append(actions[agent.firm_id][0])
            episode_demand.append(env.demand[agent.firm_id][0])
            episode_satisfied_demand.append(env.satisfied_demand[agent.firm_id][0])

            # 当前智能体的奖励
            reward = rewards[agent.firm_id][0]
            score += reward

            # 状态更新
            state = next_state

            if done:
                break

        # 记录本次 episode 的结果
        scores.append(score)
        inventory_history.append(episode_inventory)
        orders_history.append(episode_orders)
        demand_history.append(episode_demand)
        satisfied_demand_history.append(episode_satisfied_demand)

        print(f'Test Episode {i_episode}/{num_episodes} | Score: {score:.2f}')

    return scores, inventory_history, orders_history, demand_history, satisfied_demand_history


def plot_training_results(scores, window_size=100):
    """
    绘制训练过程中的奖励曲线。
    :param scores: 每个 episode 的奖励列表
    :param window_size: 移动平均窗口大小
    """
    def moving_average(data, window_size):
        return [np.mean(data[max(0, i - window_size):i + 1]) for i in range(len(data))]

    avg_scores = moving_average(scores, window_size)

    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(scores)), scores, alpha=0.3, label='原始奖励')
    plt.plot(np.arange(len(avg_scores)), avg_scores, label=f'{window_size}个episode的移动平均')
    plt.title('PPO训练过程中的奖励')
    plt.xlabel('Episode')
    plt.ylabel('奖励')
    plt.legend()
    plt.savefig('figures/training_rewards_ppo.png')
    plt.close()
    print("训练结果图已保存至 figures/training_rewards_ppo.png")


def plot_test_results(scores, inventory_history, orders_history, demand_history, satisfied_demand_history):
    """
    绘制测试结果，包括库存、订单、需求与满足需求、奖励分布。

    :param scores: 每个 episode 的奖励
    :param inventory_history: 每个 episode 的库存历史
    :param orders_history: 每个 episode 的订单历史
    :param demand_history: 每个 episode 的需求历史
    :param satisfied_demand_history: 每个 episode 的满足需求历史
    """
    avg_inventory = np.mean(inventory_history, axis=0)
    avg_orders = np.mean(orders_history, axis=0)
    avg_demand = np.mean(demand_history, axis=0)
    avg_satisfied_demand = np.mean(satisfied_demand_history, axis=0)

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # 平均库存
    axs[0, 0].plot(avg_inventory)
    axs[0, 0].set_title('平均库存')
    axs[0, 0].set_xlabel('时间步')
    axs[0, 0].set_ylabel('库存量')

    # 平均订单量
    axs[0, 1].plot(avg_orders)
    axs[0, 1].set_title('平均订单量')
    axs[0, 1].set_xlabel('时间步')
    axs[0, 1].set_ylabel('订单量')

    # 平均需求 vs 平均满足需求
    axs[1, 0].plot(avg_demand, label='需求')
    axs[1, 0].plot(avg_satisfied_demand, label='满足的需求')
    axs[1, 0].set_title('平均需求 vs 满足的需求')
    axs[1, 0].set_xlabel('时间步')
    axs[1, 0].set_ylabel('数量')
    axs[1, 0].legend()

    # 测试奖励柱状图
    axs[1, 1].bar(range(len(scores)), scores)
    axs[1, 1].set_title('测试episode奖励')
    axs[1, 1].set_xlabel('Episode')
    axs[1, 1].set_ylabel('总奖励')

    plt.tight_layout()
    plt.savefig('figures/test_results_ppo.png')
    plt.close()
    print("测试结果图已保存至 figures/test_results_ppo.png")


# ============================================================
# 主程序入口
# ============================================================
if __name__ == "__main__":
    # 创建保存模型和图表的目录
    os.makedirs('models', exist_ok=True)
    os.makedirs('figures', exist_ok=True)

    # 设置 Matplotlib 中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定中文字体
    plt.rcParams['axes.unicode_minus'] = False    # 正确显示负号

    # 初始化环境参数
    num_firms = 3              # 企业数量
    p = [10, 9, 8]             # 各企业产品价格列表
    h = 0.5                    # 库存持有成本
    c = 2                      # 损失销售成本
    initial_inventory = 100    # 初始库存
    poisson_lambda = 10        # 最下游企业需求的泊松分布均值
    max_steps = 100            # 每个 episode 的最大步数

    # 创建环境
    env = Env(num_firms, p, h, c, initial_inventory, poisson_lambda, max_steps)

    # 为第二个企业（firm_id=1）创建 PPO 智能体
    firm_id = 1
    state_size = 3             # 每个企业的状态维度：[订单, 满足需求, 库存]
    max_order = 20             # 最大订单量
    action_size = max_order + 1  # 动作空间大小：订单量 0~max_order，共 max_order+1 个动作

    agent = PPOAgent(
        state_size=state_size,
        action_size=action_size,
        firm_id=firm_id,
        max_order=max_order,
        gamma=0.99,
        lr=3e-4,
        eps_clip=0.2,
        K_epochs=3,          
        gae_lambda=0.95,
        entropy_coef=0.05,   
        min_entropy_coef=0.001,
        entropy_decay=0.995, 
        value_coef=0.5
    )

    # 训练 PPO 智能体
    scores = train_ppo(env, agent, num_episodes=2000, max_t=max_steps, update_every=5)

    # 绘制训练结果
    plot_training_results(scores)

    # 测试训练好的智能体
    test_scores, inventory_history, orders_history, demand_history, satisfied_demand_history = test_agent(env, agent, num_episodes=10)

    # 绘制测试结果
    plot_test_results(test_scores, inventory_history, orders_history, demand_history, satisfied_demand_history)
