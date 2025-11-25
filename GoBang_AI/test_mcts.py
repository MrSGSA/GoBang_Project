# test_mcts.py
import torch
from model import game_net
from rule import game_rule
from mcts import MCTS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = game_net().to(DEVICE)
model.load_state_dict(torch.load("backup/gomoku_model_epoch2000.pth", map_location=DEVICE))
model.eval()

env = game_rule()
env.reset()

with torch.no_grad():
    logits, value = model(torch.zeros(1,1,15,15).to(DEVICE))
    print("Policy logits:", logits)
    print("Value:", value)
    print("Softmax policy sum:", torch.softmax(logits, dim=1).sum())


print("Testing MCTS on empty board...")
mcts = MCTS(model, num_simulations=50, device=DEVICE)  # 先用 50 次模拟
action = mcts.run(env)
print(f"MCTS chose: {action} → ({action//15}, {action%15})")
