import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class GridEnv:
    def __init__(self, size=5):
        self.size = size
        self.goal = (size-1, size-1)
        self.start = (0, 0)
        # Add some obstacles for interest
        self.obstacles = set([(2,1), (3,2)])
        self.actions = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
        self.action_names = list(self.actions.keys())
        self.reset()
    
    def reset(self):
        self.current = self.start
        return self.current
    
    def step(self, action):
        dx, dy = self.actions[action]
        nx, ny = self.current[0] + dx, self.current[1] + dy
        if not (0 <= nx < self.size and 0 <= ny < self.size) or (nx, ny) in self.obstacles:
            # Invalid move: stay put with penalty
            reward = -1
            done = False
        else:
            self.current = (nx, ny)
            reward = -1  # Cost per step
            done = self.current == self.goal
            if done:
                reward = 10  # Bonus for reaching goal
        return self.current, reward, done

# Q-Learning parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate
episodes = 1000

# Initialize Q-table for non-obstacle states
env = GridEnv()
Q = {}
for i in range(env.size):
    for j in range(env.size):
        if (i, j) not in env.obstacles:
            Q[(i, j)] = {a: 0.0 for a in env.action_names}

def choose_action(state, Q, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.choice(env.action_names)
    else:
        # Greedy: pick action with highest Q-value
        return max(Q[state], key=Q[state].get)

def train():
    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = choose_action(state, Q, epsilon)
            next_state, reward, done = env.step(action)
            # Q-update only for valid states and actions
            if state in Q and action in Q[state] and next_state in Q:
                old_value = Q[state][action]
                next_max = max(Q[next_state].values())
                new_value = old_value + alpha * (reward + gamma * next_max - old_value)
                Q[state][action] = new_value
            state = next_state
        if episode % 200 == 0:
            print(f"Episode {episode} completed")

# Train the agent
train()

# Find the learned path using greedy policy
def find_shortest_path(Q, start, goal):
    path = [start]
    state = start
    visited = set([start])
    while state != goal:
        action = max(Q[state], key=Q[state].get)
        dx, dy = env.actions[action]
        next_state = (state[0] + dx, state[1] + dy)
        if next_state not in Q or next_state in visited:
            # Loop or invalid, stop
            break
        path.append(next_state)
        visited.add(next_state)
        state = next_state
    return path

path = find_shortest_path(Q, env.start, env.goal)
print(f"Learned shortest path: {path}")
print(f"Path length (steps): {len(path) - 1}")

# Optional: Print Q-table for a sample state
print("\nSample Q-values for state (0,0):")
print(Q[(0,0)])

# Animated visualization of the path
def animate_path(env, path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-0.5, env.size - 0.5)
    ax.set_ylim(-0.5, env.size - 0.5)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='-', alpha=0.3)
    
    # Draw grid cells
    for i in range(env.size):
        for j in range(env.size):
            if (i, j) in env.obstacles:
                ax.add_patch(plt.Rectangle((j, env.size - 1 - i), 1, 1, color='black', label='Obstacle' if (i,j)==list(env.obstacles)[0] else ""))
            else:
                ax.add_patch(plt.Rectangle((j, env.size - 1 - i), 1, 1, fill=False, edgecolor='gray'))
    
    # Plot start and goal
    start_pos = env.start
    ax.plot(start_pos[1], env.size - 1 - start_pos[0], 'go', markersize=10, label='Start')
    
    goal_pos = env.goal
    ax.plot(goal_pos[1], env.size - 1 - goal_pos[0], 'ro', markersize=10, label='Goal')
    
    # Plot full path line
    if len(path) > 1:
        path_x = [p[1] for p in path]
        path_y = [env.size - 1 - p[0] for p in path]
        ax.plot(path_x, path_y, 'b--', linewidth=1, alpha=0.5, label='Path')
    
    # Initialize agent marker
    agent, = ax.plot([], [], 'bo', markersize=8, label='Agent')
    
    ax.set_title('Animated Shortest Path in Grid World')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row (inverted for plotting)')
    ax.legend()
    
    # Animation setup
    def update(frame):
        if frame < len(path):
            pos = path[frame]
            x = pos[1]
            y = env.size - 1 - pos[0]
            agent.set_data([x], [y])
        return agent,
    
    ani = FuncAnimation(fig, update, frames=len(path), interval=500, blit=True, repeat=True)
    plt.tight_layout()
    plt.show()
    return ani

# Animate the path
ani = animate_path(env, path)