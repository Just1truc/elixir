# Frequently Asked Questions (FAQ)

This document addresses common questions and issues encountered when using the Zappy AI Training Environment.

## General Questions

### Q: What is the Zappy AI Training Environment?
**A:** The Zappy AI Training Environment is a socket-free simulation of the Zappy game designed specifically for training AI agents. It provides complete game mechanics without network overhead, making it ideal for reinforcement learning and AI development.

### Q: How does this differ from the original Zappy server?
**A:** The key differences are:
- **No network communication**: Direct Python API instead of socket-based protocol
- **Training-focused**: Built-in reward systems and environment variants
- **Performance optimized**: No I/O overhead, faster simulation speeds
- **AI-friendly**: State observation and action spaces designed for ML algorithms
- **Reproducible**: Seed-based random generation for consistent experiments

### Q: What Python version is required?
**A:** Python 3.9 or higher is required. The environment uses modern type hints and features that require recent Python versions.

## Environment Questions

### Q: How do I choose the right environment for training?
**A:** Choose based on your training objectives:

- **`alone`**: Basic movement and resource collection
- **`survival`**: Resource management under pressure
- **`group`**: Multi-agent cooperation and coordination
- **`incantation`**: Strategic planning for elevation rituals
- **`random_group`**: Handling varied skill levels and teamwork
- **`close_incantation`**: Goal-directed behavior and problem-solving

### Q: Can I create custom environments?
**A:** Yes! Inherit from the `Environment` base class and override:
```python
class MyEnvironment(Environment):
    def _setup(self):
        # Custom initialization
        pass
    
    def _compute_reward(self, player, prev_state):
        # Custom reward logic
        return reward
    
    def _check_termination(self):
        # Custom end conditions
        return is_done
```

### Q: How do I make environments reproducible?
**A:** Use the `seed` parameter:
```python
env = Category["survival"](size=10, seed=42)
# This will produce identical results across runs
```

### Q: What's the difference between `nb_connect` and `nb_eggs`?
**A:** 
- `nb_connect`: Initial connection slots (eggs) available per team
- `nb_eggs`: Used in server initialization, typically same as `nb_connect`
- These represent spawn points where new players can join teams

## Player and Commands

### Q: Why do commands take multiple ticks to execute?
**A:** This simulates the original Zappy game timing where actions have duration. Command execution times are:
- Movement commands (forward, left, right): 7 ticks
- Resource commands (take, set): 7 ticks  
- Look command: 7 ticks
- Broadcast: 7 ticks
- Fork: 42 ticks
- Incantation: 300 ticks

### Q: How do I speed up simulation for training?
**A:** You can modify command timings:
```python
from player import CommandTick, Player

# Speed up for training
CommandTick[Player.forward] = 1
CommandTick[Player.look] = 1
CommandTick[Player.take] = 1
```

### Q: Can players see other players directly?
**A:** No, players must use the `look` command to observe their current tile, or `broadcast` for communication. There's no direct visibility of other players' positions.

### Q: How does the broadcast system work?
**A:** When a player broadcasts:
1. The sender receives "message 0, <text>"
2. Other players receive "message <direction>, <text>" where direction (1-8) indicates the relative position of the sender
3. Direction calculation uses toroidal distance and octant-based mapping

### Q: What happens when a player dies?
**A:** 
- Player's `is_alive` becomes `False`
- They can no longer execute commands
- They're removed from active simulation
- Death triggers a large negative reward (-100 by default)

## Rewards and Training

### Q: How is the reward system designed?
**A:** The reward system uses a hierarchical approach:
1. **Base rewards**: Common across all environments (death, level progression, food)
2. **Environment-specific rewards**: Tailored to each scenario's objectives
3. **Behavioral rewards**: Encourage specific strategies (exploration, cooperation)

### Q: Why are some rewards negative?
**A:** Negative rewards (penalties) guide AI behavior:
- Step penalty (-0.01): Encourages efficiency
- Death penalty (-100): Strongly discourages dangerous actions  
- Starvation penalty (-0.5): Encourages food management

### Q: How do I debug reward calculation?
**A:** Add logging to the reward function:
```python
def _compute_reward(self, player, prev_state):
    reward = 0
    base_reward = super()._compute_reward(player, prev_state)
    
    print(f"Player {player.id}: Base reward = {base_reward}")
    
    # Add environment-specific rewards
    if player.level > prev_state["level"]:
        level_bonus = 10
        reward += level_bonus
        print(f"Level bonus: +{level_bonus}")
    
    total = base_reward + reward
    print(f"Total reward: {total}")
    return total
```

### Q: Can I modify rewards during training?
**A:** Yes, either through reward constants:
```python
from environment.reward_constant import RewardConstants
RewardConstants.LEVEL_PROGRESSION_BASE = 20  # Increase level rewards
```

Or by subclassing the environment with custom reward logic.

## Technical Issues

### Q: The simulation is running slowly. How can I optimize it?
**A:** Several optimization strategies:

1. **Reduce command timing**:
```python
CommandTick = {cmd: 1 for cmd in CommandTick}  # Make all commands instant
```

2. **Smaller maps**: Use `size=5` or `size=8` for faster simulation

3. **Fewer players**: Reduce `nb_players` for multi-agent environments

4. **Batch processing**: Process multiple steps at once

### Q: I'm getting memory leaks during long training runs. What should I do?
**A:** Implement periodic cleanup:
```python
def training_loop():
    env = Category["survival"]()
    
    for episode in range(10000):
        env.reset()
        # ... training logic
        
        # Cleanup every 100 episodes
        if episode % 100 == 0:
            import gc
            gc.collect()
```

### Q: How do I handle inconsistent behavior between runs?
**A:** This usually indicates randomness issues:
1. Always set a seed: `env = Category["env_type"](seed=42)`
2. Avoid global state modifications
3. Reset random state if needed: `random.seed(42)`

### Q: Tests are failing after I modified the code. What should I check?
**A:** Common issues:
1. **Command timing changes**: Tests expect specific timing values
2. **Reward modifications**: Tests may check specific reward values  
3. **Map size changes**: Some tests assume default sizes
4. **Resource density changes**: Tests may expect certain resource amounts

Run tests with: `python3 tests/tests.py -v` for detailed output.

### Q: How do I add new resource types?
**A:** 
1. Add to the `Resource` enum in `constant.py`
2. Update `RESOURCE_ENUM` mapping
3. Add to `RESOURCE_DENSITY` 
4. Update elevation requirements if needed
5. Modify map initialization to handle new resource

```python
# In constant.py
class Resource(Enum):
    # ... existing resources
    NEW_RESOURCE = "new_resource"

RESOURCE_ENUM["new_resource"] = Resource.NEW_RESOURCE
RESOURCE_DENSITY[Resource.NEW_RESOURCE] = 0.03
```

## Integration Questions

### Q: How do I integrate with popular RL frameworks (Stable-Baselines3, Ray RLlib)?
**A:** Create a Gym-style wrapper:
```python
import gym
from gym import spaces
import numpy as np

class ZappyGymWrapper(gym.Env):
    def __init__(self, env_name="survival", **kwargs):
        self.env = Category[env_name](**kwargs)
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(len(ActionSpace.ACTIONS))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(feature_size,), dtype=np.float32
        )
    
    def reset(self):
        state = self.env.reset()
        return self._state_to_obs(state)
    
    def step(self, action):
        # Convert action to command
        player = list(self.env.players.values())[0]
        ActionSpace.execute_action(player, action)
        
        state, rewards, done = self.env.step()
        obs = self._state_to_obs(state)
        reward = sum(rewards.values()) if rewards else 0
        
        return obs, reward, done, {}
```

### Q: Can I use this with multi-processing for parallel training?
**A:** Yes, but be careful with shared state:
```python
import multiprocessing as mp

def train_worker(worker_id, seed_offset):
    env = Category["survival"](seed=42 + seed_offset)
    # ... training logic
    
# Start multiple workers
processes = []
for i in range(4):
    p = mp.Process(target=train_worker, args=(i, i * 1000))
    p.start()
    processes.append(p)

for p in processes:
    p.join()
```

### Q: How do I save and load environment states?
**A:** Use the clone functionality:
```python
# Save state
saved_state = env.server.clone()

# Restore state later
env.server = saved_state
env.players = saved_state.players
env.map = saved_state.map
```

## Performance Questions

### Q: What's the expected simulation speed?
**A:** Performance depends on configuration:
- **Small environment** (5x5, 2 players): 10,000+ steps/second
- **Medium environment** (10x10, 4 players): 5,000+ steps/second  
- **Large environment** (20x20, 10 players): 1,000+ steps/second

### Q: How much memory does the environment use?
**A:** Approximate memory usage:
- **Per player**: ~1KB (state + command queues)
- **Per tile**: ~100 bytes (resource counts)
- **Base environment**: ~1MB
- **Large simulation** (50x50, 100 players): ~10MB

### Q: Can I run multiple environments simultaneously?
**A:** Yes, environments are independent:
```python
envs = [Category["survival"](seed=i) for i in range(10)]

# Run all environments
for env in envs:
    state, rewards, done = env.step()
```

This FAQ covers the most common questions. For additional help, refer to the detailed documentation or examine the test cases for usage examples.
