# Training Guide

This guide will help you understand how to train AI agents in the Zappy environment effectively.

## Overview

The Zappy AI Training Environment provides a complete simulation of the Zappy game without network overhead, making it ideal for reinforcement learning and AI development. The environment supports multiple training scenarios, each designed to teach different aspects of the game.

## Environment Selection

### 1. Alone Environment
**Purpose:** Learn basic movement and resource collection
**Use Case:** Initial training, movement mechanics, resource gathering

```python
from environment import Category

env = Category["alone"](level=1, size=10)
state = env.reset()

# Train on basic survival and resource collection
for episode in range(1000):
    state = env.reset()
    done = False
    
    while not done:
        # Simple policy: collect food when low, explore otherwise
        player = list(state["players"].values())[0]
        
        if player.inventory["food"] < 5:
            # Look for food
            player.add_cmd("Look")
        else:
            # Random exploration
            action = random.choice(["Forward", "Left", "Right"])
            player.add_cmd(action)
        
        state, rewards, done = env.step()
```

### 2. Survival Environment
**Purpose:** Learn resource management under pressure
**Use Case:** Food scarcity scenarios, survival optimization

```python
env = Category["survival"](nb_players=4, food_amount=2, size=8)

# Survival training emphasizes efficient food collection
def survival_policy(player, tile_info):
    food_level = player.inventory["food"]
    
    if food_level < 2:
        # Critical - must find food immediately
        return prioritize_food_search(player, tile_info)
    elif food_level < 5:
        # Low - balance food search with exploration
        return balanced_food_exploration(player, tile_info)
    else:
        # Safe - can focus on other objectives
        return explore_or_gather_resources(player, tile_info)
```

### 3. Group Environment
**Purpose:** Learn cooperation and communication
**Use Case:** Multi-agent coordination, team strategies

```python
env = Category["group"](level=2, nb_players=3, size=10)

# Group training focuses on coordination
def group_policy(players, state):
    for player_id, player in players.items():
        # Check if enough players for incantation
        same_tile_players = count_players_on_tile(player.position, players)
        required_players = ElevationRequirement.requirements[player.level][0]
        
        if same_tile_players >= required_players:
            # Attempt incantation
            player.add_cmd("Incantation")
        else:
            # Broadcast location to gather teammates
            player.add_cmd(f"Broadcast {player.position[0]},{player.position[1]}")
            # Move toward other players
            move_toward_teammates(player, players)
```

## Training Strategies

### Reinforcement Learning Setup

#### State Representation
```python
def get_state_vector(player, map_info):
    """Convert game state to feature vector for RL algorithms"""
    features = []
    
    # Player features
    features.extend([
        player.position[0] / map_size,  # Normalized X
        player.position[1] / map_size,  # Normalized Y
        player.direction.value / 4,     # Normalized direction
        player.level / 8,               # Normalized level
    ])
    
    # Inventory features (normalized)
    for resource in Resource:
        features.append(player.inventory[resource.value] / 10)
    
    # Local environment (from look command)
    tile_resources = get_tile_resources(player.position, map_info)
    for resource in Resource:
        features.append(tile_resources.get(resource, 0) / 5)
    
    # Team information
    features.append(len(get_teammates_nearby(player)) / 10)
    
    return np.array(features)
```

#### Action Space Design
```python
class ActionSpace:
    """Discrete action space for RL agents"""
    ACTIONS = [
        "Forward",
        "Left", 
        "Right",
        "Look",
        "Take food",
        "Take linemate",
        "Take deraumere",
        "Take sibur",
        "Take mendiane", 
        "Take phiras",
        "Take thystame",
        "Set food",
        "Set linemate",
        # ... other set actions
        "Fork",
        "Eject",
        "Incantation",
        "Broadcast help",
        "Broadcast position"
    ]
    
    @staticmethod
    def execute_action(player, action_id):
        action = ActionSpace.ACTIONS[action_id]
        player.add_cmd(action)
```

### Deep Q-Learning Implementation

```python
import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random

class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size, lr=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        self.q_network = DQN(state_size, action_size)
        self.target_network = DQN(state_size, action_size)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        
    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.choice(range(self.action_size))
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
            
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_network(next_states).max(1)[0].detach()
        target_q = rewards + (0.99 * next_q * ~dones)
        
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
```

### Training Loop

```python
def train_agent(env_name, episodes=5000):
    env = Category[env_name](size=10)
    
    # Get state and action dimensions
    dummy_state = env.reset()
    dummy_player = list(dummy_state["players"].values())[0]
    state_size = len(get_state_vector(dummy_player, dummy_state))
    action_size = len(ActionSpace.ACTIONS)
    
    agent = DQNAgent(state_size, action_size)
    scores = deque(maxlen=100)
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        
        while True:
            # Get current player
            player = list(state["players"].values())[0]
            current_state = get_state_vector(player, state)
            
            # Choose action
            action = agent.act(current_state)
            ActionSpace.execute_action(player, action)
            
            # Step environment
            next_state, rewards, done = env.step()
            reward = sum(rewards.values()) if rewards else 0
            
            # Get next state vector
            next_player = list(next_state["players"].values())[0]
            next_state_vector = get_state_vector(next_player, next_state)
            
            # Store experience
            agent.remember(current_state, action, reward, next_state_vector, done)
            
            # Update state
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        # Train agent
        agent.replay()
        scores.append(total_reward)
        
        # Update target network periodically
        if episode % 100 == 0:
            agent.target_network.load_state_dict(agent.q_network.state_dict())
            
        # Print progress
        if episode % 100 == 0:
            avg_score = np.mean(scores)
            print(f"Episode {episode}, Average Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.3f}")
```

## Advanced Training Techniques

### Curriculum Learning
Start with simple environments and gradually increase complexity:

```python
curriculum = [
    {"env": "alone", "episodes": 1000, "params": {"level": 1, "size": 5}},
    {"env": "alone", "episodes": 1000, "params": {"level": 2, "size": 8}},
    {"env": "survival", "episodes": 2000, "params": {"nb_players": 2, "food_amount": 3, "size": 8}},
    {"env": "group", "episodes": 3000, "params": {"level": 2, "nb_players": 3, "size": 10}},
    {"env": "incantation", "episodes": 2000, "params": {"level": 3, "size": 12}},
]

for stage in curriculum:
    print(f"Training on {stage['env']} environment...")
    train_agent_curriculum(stage["env"], stage["episodes"], stage["params"])
```

### Multi-Agent Training
```python
def train_multi_agent(env_name, num_agents=4):
    env = Category[env_name](nb_players=num_agents, size=12)
    agents = [DQNAgent(state_size, action_size) for _ in range(num_agents)]
    
    for episode in range(episodes):
        state = env.reset()
        
        while True:
            # Each agent acts independently
            for i, (player_id, player) in enumerate(state["players"].items()):
                if player.is_alive:
                    current_state = get_state_vector(player, state)
                    action = agents[i].act(current_state)
                    ActionSpace.execute_action(player, action)
            
            next_state, rewards, done = env.step()
            
            # Update each agent
            for i, (player_id, reward) in enumerate(rewards.items()):
                if player_id in state["players"]:
                    # Store experience for this agent
                    # ... update logic
            
            state = next_state
            if done:
                break
```

## Performance Monitoring

### Metrics to Track
- **Survival Rate**: Percentage of episodes where player survives
- **Level Progression**: Average level reached per episode
- **Resource Efficiency**: Resources collected vs. time spent
- **Cooperation Score**: Successful team interactions (for group environments)

### Visualization
```python
import matplotlib.pyplot as plt

def plot_training_progress(scores, levels, survival_rates):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Reward progression
    ax1.plot(scores)
    ax1.set_title('Training Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    
    # Level progression
    ax2.plot(levels)
    ax2.set_title('Level Progression')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Max Level Reached')
    
    # Survival rate
    ax3.plot(survival_rates)
    ax3.set_title('Survival Rate')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Survival Rate (%)')
    
    plt.tight_layout()
    plt.show()
```

## Troubleshooting

### Common Training Issues

1. **Agent Not Learning**
   - Check reward signal strength
   - Verify state representation includes necessary information
   - Adjust exploration rate (epsilon) decay

2. **Overfitting to Environment**
   - Use curriculum learning
   - Randomize environment parameters
   - Add noise to observations

3. **Poor Multi-Agent Coordination**
   - Implement communication rewards
   - Use centralized training with decentralized execution
   - Add team-based reward shaping

### Debugging Tools
```python
def debug_episode(env, agent, max_steps=100):
    """Run single episode with detailed logging"""
    state = env.reset()
    step = 0
    
    while step < max_steps:
        player = list(state["players"].values())[0]
        print(f"Step {step}: Position {player.position}, Level {player.level}")
        print(f"Inventory: {player.inventory}")
        
        current_state = get_state_vector(player, state)
        action = agent.act(current_state)
        print(f"Action: {ActionSpace.ACTIONS[action]}")
        
        ActionSpace.execute_action(player, action)
        state, rewards, done = env.step()
        
        print(f"Reward: {sum(rewards.values()) if rewards else 0}")
        print("---")
        
        if done:
            break
        step += 1
```

This training guide provides a comprehensive foundation for developing AI agents in the Zappy environment. Start with simple scenarios and gradually increase complexity as your agents improve.
