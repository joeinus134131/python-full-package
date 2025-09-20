import gymnasium as gym
from stable_baselines3 import DQN

# 1. Buat environment RL
env = gym.make("CartPole-v1", render_mode="human")

# 2. Buat dan latih model DQN
model = DQN(
    policy="MlpPolicy",      # pakai neural network (MLP)
    env=env,                 
    learning_rate=1e-3,      
    buffer_size=10000,       
    learning_starts=1000,    
    batch_size=32,
    gamma=0.99,               # faktor diskon reward masa depan
    verbose=1
)

# 3. Latih agent-nya (10000 langkah)
model.learn(total_timesteps=10_000)

# 4. Jalankan agent yang sudah terlatih
obs, _ = env.reset()
done = False
total_reward = 0

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    done = terminated or truncated

print(f"🎯 Episode selesai dengan total reward: {total_reward}")
env.close()
