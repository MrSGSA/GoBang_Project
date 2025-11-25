# mcts.py
import torch
import numpy as np
import copy
import math


class TreeNode:
    def __init__(self, parent, prior_p):
        self.parent = parent
        self.children = {}  # {action: TreeNode}
        self.n_visits = 0
        self.Q = 0
        self.u = 0
        self.P = prior_p

    def expand(self, action_priors):
        """扩展子节点"""
        for action, prob in action_priors:
            if action not in self.children:
                self.children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """
        选择 UCB 值最大的子节点
        UCB = Q + c_puct * P * sqrt(parent_visits) / (1 + child_visits)
        """
        return max(self.children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def get_value(self, c_puct):
        """计算 Upper Confidence Bound (UCB)"""
        # 加上 1e-10 防止除以 0（虽然通常 n_visits 初始为0时公式能处理，但加保险）
        self.u = (c_puct * self.P * math.sqrt(self.parent.n_visits) / (1 + self.n_visits))
        return self.Q + self.u

    def update(self, leaf_value):
        """
        反向传播更新价值
        leaf_value: 对于当前节点所属玩家的价值 (v)
        """
        self.n_visits += 1
        # Q 值更新：累计平均
        self.Q += (leaf_value - self.Q) / self.n_visits

        # 递归更新父节点
        # 注意：父节点是对手，所以价值取反 (-leaf_value)
        if self.parent:
            self.parent.update(-leaf_value)

    def is_leaf(self):
        return len(self.children) == 0


class MCTS:
    def __init__(self, policy_value_fn, c_puct=5, num_simulations=100, device="cpu"):
        self.policy_value_fn = policy_value_fn  # 神经网络模型
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.device = device
        self.root = None

    def _playout(self, env):
        """执行一次模拟：Selection -> Expansion -> Evaluation -> Backup"""
        node = self.root

        # 1. Selection: 一直走到叶子节点
        while not node.is_leaf():
            action, node = node.select(self.c_puct)
            env.step(action)

        # 此时 env 处于叶子节点状态
        # 2. 判断游戏是否结束
        winner, is_end = env.has_a_winner()  # 建议 rule.py 统一返回 (winner, end_flag)
        # 如果你的 rule.py 只有 env.winner 和 env.board，可以用下面的逻辑：
        # winner = env.winner
        # is_end = (winner != 0) or np.all(env.board != 0)

        if is_end:
            if winner == 0:
                leaf_value = 0.0
            else:
                # 游戏结束且有赢家。
                # 由于这是通过 step 进入的节点，说明上一步走棋的人赢了。
                # 也就是当前节点对应的玩家（等待落子的人）输了。
                # 所以对于当前节点玩家，价值是 -1。
                leaf_value = -1.0

                # 反向传播 (注意：update内部会自动处理父节点的取反)
            node.update(leaf_value)
            return

        # 3. Expansion & Evaluation (通过神经网络)

        # --- 🔥 关键修改：视角转换 (Canonical Form) ---
        # 必须把盘面转换成“当前玩家是黑棋(1)”的视角
        # 假设 steps 长度为偶数是黑棋回合，奇数是白棋回合
        current_player = 1 if len(env.steps) % 2 == 0 else -1
        canonical_board = env.board * current_player

        # 转换为 Tensor
        state_tensor = torch.from_numpy(canonical_board).float().to(self.device).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            # 假设模型输出是 log_softmax(act) 和 tanh(val)
            log_action_probs, leaf_value = self.policy_value_fn(state_tensor)

        # 处理 Value
        leaf_value = leaf_value.item()  # [-1, 1]

        # 处理 Policy
        # 因为模型输出是 log_softmax，我们需要 exp 变回概率
        action_probs = np.exp(log_action_probs.cpu().numpy().flatten())

        # 获取合法动作
        valid_actions = env.get_valid_actions()

        # 过滤并归一化概率
        probs = action_probs[valid_actions]
        probs_sum = np.sum(probs)
        if probs_sum > 0:
            probs /= probs_sum  # 重新归一化
        else:
            # 极罕见情况：模型认为所有合法动作概率都极小，退化为均匀分布
            probs = np.ones(len(valid_actions)) / len(valid_actions)

        # 扩展节点
        action_priors = zip(valid_actions, probs)
        node.expand(action_priors)

        # 4. Backup
        node.update(leaf_value)

    def run(self, env, temp=1.0):
        """
        执行 MCTS 搜索并返回动作
        temp: 温度参数
        """
        self.root = TreeNode(None, 1.0)

        # --- 🔥 优化：增加根节点噪声 (Dirichlet Noise) ---
        # 仅在训练阶段(temp > 0)或确实需要探索时添加
        # 这有助于防止模型在自我博弈中过早收敛到单一策略
        if temp > 0:
            # 先跑一次网络获取 Prior，以便加噪声
            # 注意：这里为了简化，我们通常依赖第一次 simulation 来展开根节点
            # 但为了加噪声，我们需要确保根节点已经展开。

            # 简单做法：先做一次模拟，确保 root 展开
            sim_env = copy.deepcopy(env)
            self._playout(sim_env)

            # 添加噪声
            if self.root.children:
                actions = list(self.root.children.keys())
                noise = np.random.dirichlet([0.3] * len(actions))
                epsilon = 0.25  # 噪声权重，AlphaZero 标准是 0.25

                for i, action in enumerate(actions):
                    node = self.root.children[action]
                    # 混合网络预测概率(P)与噪声
                    node.P = (1 - epsilon) * node.P + epsilon * noise[i]

        # 开始正式模拟
        for _ in range(self.num_simulations):
            simulation_env = copy.deepcopy(env)
            self._playout(simulation_env)

        # 计算每个动作的访问次数
        counts = [(act, node.n_visits) for act, node in self.root.children.items()]

        if not counts:
            # 异常保护：如果没有合法动作（虽然 playout 应该处理了）
            return np.random.choice(env.get_valid_actions())

        acts, visits = zip(*counts)
        visits = np.array(visits)

        if temp == 0:
            # 评估/竞技模式：贪婪选择访问量最大的
            action = acts[np.argmax(visits)]
        else:
            # 训练模式：根据访问量概率分布采样
            # 为了数值稳定性
            if temp == 1.0:
                probs = visits / np.sum(visits)
            else:
                # 避免过大的 temp 导致溢出，或者过小的 temp 导致除零
                visits = visits ** (1.0 / temp)
                probs = visits / np.sum(visits)

            action = np.random.choice(acts, p=probs)

        return action

    def reset_player(self):
        self.root = TreeNode(None, 1.0)

    def set_simulations(self, n):
        self.num_simulations = n