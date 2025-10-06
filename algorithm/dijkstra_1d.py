import numpy as np
import random

# Simple 1D line graph: nodes 0 -> 1 -> 2 -> 3 -> 4
# Edges: 0-1 (weight 1), 1-2 (weight 1), 2-3 (weight 1), 3-4 (weight 1)
# Goal: reach node 4 from node 0
# We'll use Q-learning to learn the shortest path policy

class LineGraphEnv:
    def __init__(self):
        self.nodes = 5  # nodes 0 to 4
        self.goal = 4
        self.adjacency = {
            0: {1: 1},
            1: {0: 1, 2: 1},
            2: {1: 1, 3: 1},
            3: {2: 1, 4: 1},
            4: {}
        }
        self.reset()
    
    def reset(self):
        self.current_state = 0
        return self.current_state
    
    def step(self, action):
        next_state = action
        if next_state not in self.adjacency[self.current_state]:
            # Invalid action, stay and penalize
            reward = -10
            done = False
        else:
            cost = self.adjacency[self.current_state][next_state]
            reward = -cost
            self.current_state = next_state
            done = (self.current_state == self.goal)
            if done:
                reward += 10  # Bonus for reaching goal
        return self.current_state, reward, done

# Q-Learning parameters
num_states = 5
num_actions_per_state = 2  # Roughly, but varies; we'll map actions to neighbors
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate
episodes = 1000

# Initialize Q-table: states x possible actions (we'll use a dict of dicts for simplicity)
Q = {state: {neighbor: 0.0 for neighbor in adj} for state, adj in LineGraphEnv().adjacency.items()}

env = LineGraphEnv()

def choose_action(state, Q, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.choice(list(Q[state].keys()))
    else:
        return max(Q[state], key=Q[state].get)

def train():
    for episode in range(episodes):
        print(f"Episode {episode} starting")
        state = env.reset()
        print(f"ini state {state} data")
        done = False
        while not done:
            action = choose_action(state, Q, epsilon)
            next_state, reward, done = env.step(action)
            print(f"State: {state}, Action: {action}, Reward: {reward}, Next State: {next_state}")
            print(Q[state])
            # Q-update
            if next_state in Q[state]:  # Only update if valid transition
                old_value = Q[state][action]
                print(old_value)
                next_max = max(Q[next_state].values()) if Q[next_state] else 0
                new_value = old_value + alpha * (reward + gamma * next_max - old_value)
                Q[state][action] = new_value
            state = next_state
        if episode % 200 == 0:
            print(f"Episode {episode} completed")

# Train the agent
train()

# Test the learned policy: find path from 0 to 4
def find_shortest_path(Q, start=0, goal=4):
    path = [start]
    state = start
    while state != goal:
        action = max(Q[state], key=Q[state].get)
        path.append(action)
        state = action
    total_cost = sum(env.adjacency[path[i]][path[i+1]] for i in range(len(path)-1))
    return path, total_cost

path, cost = find_shortest_path(Q)
print(f"Learned shortest path: {path}")
print(f"Total cost: {cost}")

# Output the Q-table for inspection
print("\nQ-Table:")
for state in Q:
    print(f"State {state}: {Q[state]}")