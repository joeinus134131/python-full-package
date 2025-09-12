import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")
obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminate, truncate, info = env.step(action)
    if terminate or truncate:
        obs, info = env.reset()

env.close()