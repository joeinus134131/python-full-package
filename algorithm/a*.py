import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import heapq

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

    def is_valid_pos(self, pos):
        x, y = pos
        return 0 <= x < self.size and 0 <= y < self.size and pos not in self.obstacles

    def get_neighbors(self, pos):
        neighbors = []
        for action in self.action_names:
            dx, dy = self.actions[action]
            nx, ny = pos[0] + dx, pos[1] + dy
            if self.is_valid_pos((nx, ny)):
                neighbors.append((nx, ny))
        return neighbors

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

rl_path = find_shortest_path(Q, env.start, env.goal)
print(f"RL Learned shortest path: {rl_path}")
print(f"RL Path length (steps): {len(rl_path) - 1}")

# Optional: Print Q-table for a sample state
print("\nSample Q-values for state (0,0):")
print(Q[(0,0)])

# A* Implementation
def heuristic(a, b):
    """Manhattan distance heuristic"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star(start, goal, env):
    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start))  # (f, g, pos)
    
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    
    while open_set:
        current_f, current_g, current = heapq.heappop(open_set)
        
        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path
        
        for neighbor in env.get_neighbors(current):
            tentative_g = g_score[current] + 1  # Cost of 1 per move
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], tentative_g, neighbor))
    
    return []  # No path found

a_star_path = a_star(env.start, env.goal, env)
print(f"A* shortest path: {a_star_path}")
print(f"A* Path length (steps): {len(a_star_path) - 1}")

# Static visualization of both paths
def visualize_paths(env, rl_path, a_star_path):
    fig, ax = plt.subplots(figsize=(8, 8))
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
    
    # Plot RL path
    if len(rl_path) > 1:
        rl_x = [p[1] for p in rl_path]
        rl_y = [env.size - 1 - p[0] for p in rl_path]
        ax.plot(rl_x, rl_y, 'b-', linewidth=2, label='RL Path')
        ax.plot(rl_x, rl_y, 'bo', markersize=5)
    
    # Plot A* path
    if len(a_star_path) > 1:
        astar_x = [p[1] for p in a_star_path]
        astar_y = [env.size - 1 - p[0] for p in a_star_path]
        ax.plot(astar_x, astar_y, 'r--', linewidth=2, label='A* Path')
        ax.plot(astar_x, astar_y, 'ro', markersize=5)
    
    ax.set_title('Comparison: RL Learned Path vs A* Optimal Path')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row (inverted for plotting)')
    ax.legend()
    plt.tight_layout()
    plt.show()

# Visualize both paths
visualize_paths(env, rl_path, a_star_path)

# Animated visualization of the A* path (similar to RL animation)
def animate_path(env, path, title='Animated Path', color='b'):
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
        ax.plot(path_x, path_y, f'{color}--', linewidth=1, alpha=0.5, label='Path')
    
    # Initialize agent marker
    agent, = ax.plot([], [], f'{color}o', markersize=8, label='Agent')
    
    ax.set_title(title)
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

# Animate the A* path
print("\nAnimating A* path...")
ani_astar = animate_path(env, a_star_path, 'Animated A* Shortest Path in Grid World', 'r')