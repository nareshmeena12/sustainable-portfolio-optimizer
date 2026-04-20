import pickle
import sys
sys.path.append(".")

from env.portfolio_env import PortfolioEnv

with open("data/processed/train_data.pkl", "rb") as f:
    train_data = pickle.load(f)

env = PortfolioEnv(train_data, sector="tech")
obs, _ = env.reset()

print("Obs shape:", obs.shape)

for i in range(5):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        break