import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")

observation, info = env.reset()

print(f"Starting observation: {observation}")

episode_over = False
total_reward = 0


while not episode_over:
    action = env.action_space.sample()

    observation, reward, terminate, truncate, info = env.step(action)

    total_reward += reward

    episode_over = terminate or truncate

print(f"Episode Finished! Total reward: {total_reward}")

env.close()