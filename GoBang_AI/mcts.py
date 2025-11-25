# mcts.py
import torch
import numpy as np
import copy
import math


class TreeNode:
    def __init__(self, parent, prior_p):
        self.parent = parent
        self.children = {}
        self.n_visits = 0
        self.Q = 0
        self.u = 0
        self.P = prior_p

    def expand(self, action_priors):
        for action, prob in action_priors:
            if action not in self.children:
                self.children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        return max(self.children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def get_value(self, c_puct):
        self.u = (c_puct * self.P * math.sqrt(self.parent.n_visits) / (1 + self.n_visits))
        return self.Q + self.u

    def update(self, leaf_value):
        self.n_visits += 1
        self.Q += (leaf_value - self.Q) / self.n_visits
        if self.parent:
            self.parent.update(-leaf_value)

    def is_leaf(self):
        return len(self.children) == 0


class MCTS:
    def __init__(self, policy_value_fn, c_puct=5, num_simulations=100, device="cpu"):
        self.policy_value_fn = policy_value_fn
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.device = device
        self.root = TreeNode(None, 1.0)  # 初始化根节点

    def set_simulations(self, n):
        """动态调整模拟次数"""
        self.num_simulations = n

    def reset_player(self):
        """每局开始前重置搜索树"""
        self.root = TreeNode(None, 1.0)

    def _playout(self, env):
        node = self.root
        while not node.is_leaf():
            action, node = node.select(self.c_puct)
            env.step(action)

        winner, is_end = env.has_a_winner()

        if is_end:
            # 游戏结束，winner 是上一步走棋的人（即当前节点的 parent）
            # 对于当前等待落子的人来说，如果 winner != 0，说明他输了 -> -1
            if winner == 0:
                leaf_value = 0.0
            else:
                leaf_value = -1.0
            node.update(leaf_value)
            return

        # --- 视角转换 (Canonical Form) ---
        current_player = 1 if len(env.steps) % 2 == 0 else -1
        canonical_board = env.board * current_player

        state_tensor = torch.from_numpy(canonical_board).float().to(self.device).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            # 这里对应 model.py 输出的 log_softmax
            log_action_probs, leaf_value = self.policy_value_fn(state_tensor)

        leaf_value = leaf_value.item()
        # 将 log_softmax 还原为概率
        action_probs = np.exp(log_action_probs.cpu().numpy().flatten())

        valid_actions = env.get_valid_actions()
        probs = action_probs[valid_actions]

        # 归一化
        probs_sum = np.sum(probs)
        if probs_sum > 0:
            probs /= probs_sum
        else:
            probs = np.ones(len(valid_actions)) / len(valid_actions)

        node.expand(zip(valid_actions, probs))
        node.update(leaf_value)

    def get_action(self, env, temp=1e-3, return_prob=False):
        """
        🔥 核心修改：将原本的 run 改名为 get_action，并支持返回概率分布
        """
        # 1. 如果是训练阶段(temp>0)，添加根节点噪声
        if temp > 0 and self.root.is_leaf():
            # 确保根节点已展开
            sim_env = copy.deepcopy(env)
            self._playout(sim_env)

            if self.root.children:
                actions = list(self.root.children.keys())
                noise = np.random.dirichlet([0.3] * len(actions))
                epsilon = 0.25
                for i, action in enumerate(actions):
                    self.root.children[action].P = (1 - epsilon) * self.root.children[action].P + epsilon * noise[i]

        # 2. 执行模拟
        for _ in range(self.num_simulations):
            simulation_env = copy.deepcopy(env)
            self._playout(simulation_env)

        # 3. 统计访问次数
        act_visits = [(act, node.n_visits) for act, node in self.root.children.items()]
        if not act_visits:
            # 异常兜底：随机落子
            action = np.random.choice(env.get_valid_actions())
            probs = np.zeros(env.width * env.height)
            probs[action] = 1.0
            return (action, probs) if return_prob else action

        acts, visits = zip(*act_visits)
        visits = np.array(visits)

        # 4. 计算概率分布
        if temp == 0:
            # 贪婪模式 (Evaluation)
            best_idx = np.argmax(visits)
            action = acts[best_idx]
            # 构造 one-hot 概率（为了格式统一）
            act_probs = np.zeros(len(acts))
            act_probs[best_idx] = 1.0
        else:
            # 采样模式 (Self-play)
            # 防止 temp 过小导致溢出
            if temp < 1e-3:
                temp = 1e-3

                # 使用 softmax 风格的温度调节，或者直接 visits^(1/temp)
            # AlphaZero 标准做法是 visits^(1/temp) / sum
            visits_temp = visits ** (1.0 / temp)
            act_probs = visits_temp / np.sum(visits_temp)
            action = np.random.choice(acts, p=act_probs)

        # 5. MCTS 树复用 (Move root)
        if action in self.root.children:
            self.root = self.root.children[action]
            self.root.parent = None
        else:
            self.root = TreeNode(None, 1.0)

        # 6. 返回结果
        if return_prob:
            # 构造完整的 15x15 概率向量 (225,)
            full_probs = np.zeros(env.width * env.height)
            full_probs[list(acts)] = act_probs
            return action, full_probs
        else:
            return action