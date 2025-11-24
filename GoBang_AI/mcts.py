# mcts.py
import numpy as np
import torch
from rule import game_rule

def _is_game_over(env):
    """判断游戏是否结束（胜/负/平局）"""
    return env.winner != 0 or np.all(env.board != 0)

class MCTSNode:
    __slots__ = ('parent', 'prior_prob', 'player', 'children', 'visit_count', 'value_sum')
    def __init__(self, parent, prior_prob, player):
        self.parent = parent
        self.prior_prob = prior_prob
        self.player = player
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0

    def value(self):
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0

    def is_leaf(self):
        return len(self.children) == 0

class MCTS:
    def __init__(self, model, c_puct=1.0, num_simulations=400, device=None):
        self.model = model
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.device = device or next(model.parameters()).device

    @torch.no_grad()
    def _evaluate(self, board, current_player):
        state_tensor = torch.from_numpy(board).float().unsqueeze(0).unsqueeze(0).to(self.device)
        policy_logits, value_pred = self.model(state_tensor)
        policy = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
        value = value_pred.item()
        return policy, value

    def run(self, env):
        root = MCTSNode(None, 0.0, env.current_player)
        policy, _ = self._evaluate(env.board, env.current_player)
        valid_actions = env.get_valid_actions()
        mask = np.zeros_like(policy); mask[valid_actions] = 1.0
        policy *= mask
        if policy.sum() > 0:
            policy /= policy.sum()
        else:
            policy[valid_actions] = 1.0 / len(valid_actions)

        for action in valid_actions:
            root.children[action] = MCTSNode(root, policy[action], -env.current_player)

        for _ in range(self.num_simulations):
            node = root
            search_env = game_rule(env.size)
            search_env.board = env.board.copy()
            search_env.current_player = env.current_player
            search_env.winner = env.winner

            # ===== Select =====
            while not node.is_leaf() and not _is_game_over(search_env):
                action, node = self._select(node)
                search_env.step(action)

            # ===== Expand & Evaluate =====
            if not _is_game_over(search_env):
                policy, value = self._evaluate(search_env.board, search_env.current_player)
                valid_actions = search_env.get_valid_actions()
                mask = np.zeros_like(policy); mask[valid_actions] = 1.0
                policy *= mask
                if policy.sum() > 0:
                    policy /= policy.sum()
                else:
                    policy[valid_actions] = 1.0 / len(valid_actions)

                for action in valid_actions:
                    node.children[action] = MCTSNode(node, policy[action], -search_env.current_player)
                final_value = value if search_env.current_player == env.current_player else -value
            else:
                if search_env.winner != 0:
                    final_value = 1.0 if search_env.winner == env.current_player else -1.0
                else:
                    final_value = 0.0  # 平局

            # ===== Backup =====
            temp_node = node
            temp_value = final_value
            while temp_node is not None:
                temp_node.visit_count += 1
                temp_node.value_sum += temp_value
                temp_value = -temp_value
                temp_node = temp_node.parent

        if not root.children:
            return env.get_valid_actions()[0]
        return max(root.children.items(), key=lambda x: x[1].visit_count)[0]

    def _select(self, node):
        best_score = -float('inf')
        best_action = None
        best_child = None
        for action, child in node.children.items():
            ucb = self.c_puct * child.prior_prob * np.sqrt(node.visit_count + 1e-8) / (1 + child.visit_count)
            score = -child.value() + ucb
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        if best_child is None:
            action = next(iter(node.children))
            best_child = node.children[action]
            best_action = action
        return best_action, best_child
