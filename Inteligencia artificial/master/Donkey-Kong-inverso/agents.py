import numpy as np
import random
from collections import defaultdict

def monte_carlo_control(env, num_episodes=3000, gamma=0.99, epsilon_start=0.2, decay=0.999, min_epsilon=0.01):
    Q = defaultdict(lambda: np.zeros(4))
    returns_sum = defaultdict(lambda: 0.0)
    returns_count = defaultdict(lambda: 0.0)
    
    rewards_history, steps_history, success_history = [], [], []
    epsilon = epsilon_start
    
    for ep in range(num_episodes):
        if (ep + 1) % 500 == 0:
            print(f"   -> Procesando episodio {ep + 1}/{num_episodes}...")
            
        state = env.reset()
        done = False
        episode = []
        steps = 0  
        
        while not done and steps < 200:  # Límite de seguridad
            s_idx = env.state_to_idx(state)
            if random.random() < epsilon:
                action = random.randint(0, 3)
            else:
                action = np.argmax(Q[s_idx])
                
            next_state, reward, done = env.step(action)
            episode.append((s_idx, action, reward))
            state = next_state
            steps += 1
            
        rewards_history.append(sum([x[2] for x in episode]))
        steps_history.append(len(episode))
        success_history.append(1 if state == env.goal else 0)
        
        G = 0
        visited = set()
        for s_idx, action, reward in reversed(episode):
            G = reward + gamma * G
            if (s_idx, action) not in visited:
                visited.add((s_idx, action))
                returns_sum[(s_idx, action)] += G
                returns_count[(s_idx, action)] += 1
                Q[s_idx][action] = returns_sum[(s_idx, action)] / returns_count[(s_idx, action)]
                
        epsilon = max(min_epsilon, epsilon * decay)
        
    return Q, rewards_history, steps_history, success_history


def q_learning(env, num_episodes=3000, alpha=0.1, gamma=0.99, epsilon_start=0.2, decay=0.999, min_epsilon=0.01):
    Q = defaultdict(lambda: np.zeros(4))
    rewards_history, steps_history, success_history = [], [], []
    epsilon = epsilon_start
    
    for ep in range(num_episodes):
        if (ep + 1) % 500 == 0:
            print(f"   -> Procesando episodio {ep + 1}/{num_episodes}...")
            
        state = env.reset()
        done = False
        total_reward, steps = 0, 0
        
        while not done and steps < 200:  # <--- AGREGADO AQUÍ TAMBIÉN PARA EVITAR ATASCOS
            s_idx = env.state_to_idx(state)
            if random.random() < epsilon:
                action = random.randint(0, 3)
            else:
                action = np.argmax(Q[s_idx])
                
            next_state, reward, done = env.step(action)
            s_next_idx = env.state_to_idx(next_state)
            
            best_next_action = np.argmax(Q[s_next_idx])
            td_target = reward + (gamma * Q[s_next_idx][best_next_action] if not done else 0)
            Q[s_idx][action] += alpha * (td_target - Q[s_idx][action])
            
            state = next_state
            total_reward += reward
            steps += 1
            
        rewards_history.append(total_reward)
        steps_history.append(steps)
        success_history.append(1 if state == env.goal else 0)
        
        epsilon = max(min_epsilon, epsilon * decay)
        
    return Q, rewards_history, steps_history, success_history


def show_policy(Q, env):
    actions_str = [' ↑ ', ' ↓ ', ' ← ', ' → ']
    for r in range(env.rows):
        row_str = ""
        for c in range(env.cols):
            if (r, c) == env.goal:
                row_str += "  G  "
            elif (r, c) in env.holes:
                row_str += "  H  "
            else:
                s_idx = env.state_to_idx((r, c))
                if s_idx not in Q or np.all(Q[s_idx] == 0):
                    row_str += "  .  "
                else:
                    best_a = np.argmax(Q[s_idx])
                    row_str += actions_str[best_a]
        print(row_str)