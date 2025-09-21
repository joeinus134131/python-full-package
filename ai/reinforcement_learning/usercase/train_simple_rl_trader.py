import yfinance as yf
import pandas as pd
import gymnasium as gym
import gym_anytrading
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from gym_anytrading.envs import TradingEnv, ForexEnv, StocksEnv, Actions, Positions 
import numpy as np

symbol = "BTC-USD"
df = yf.download(symbol, start="2021-01-01", end="2023-01-01", progress=False)
print(df.head())
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
df = df.reset_index()

class MySimpleStockEnv(StocksEnv):
    _process_data = StocksEnv._process_data

    def __init__(self, df):
        super().__init__(df=df, window_size=10, frame_bound=(10, len(df)))

env = DummyVecEnv([lambda: MySimpleStockEnv(df)])
model = DQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=1e-4,
    buffer_size=50000,
    learning_starts=1000,
    batch_size=64,
    gamma=0.99,
    verbose=1
)
model.learn(total_timesteps=20_000)

obs = env.reset()
done = False
total_reward = 0    
for _ in range(500):
    action, _states= model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    # env.render()
    total_reward += rewards[0]
    if dones[0]:
        break

print(f"🎯 Episode selesai dengan total reward:", total_reward)



