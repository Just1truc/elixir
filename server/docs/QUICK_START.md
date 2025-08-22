# Quick Start Guide

Get up and running with the Zappy AI Training Environment in just a few minutes.

## Installation

### Prerequisites
- Python 3.9 or higher
- Git (for cloning the repository)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd zappy-python-server

# Optional: Create virtual environment
python3 -m venv zappy-env
source zappy-env/bin/activate  # On Windows: zappy-env\Scripts\activate

# Verify installation
python3 tests/tests.py
```

If all tests pass, you're ready to go! 🎉

## First Steps

### 1. Basic Simulation

Let's start with a simple server simulation:

```python
# simple_demo.py
from server import Server

# Create a small world
server = Server(size=10, nb_eggs=3, teams=["team1", "team2"])

# Add some players
player1 = server.add_player("team1")
player2 = server.add_player("team1")

print(f"Player 1 position: {player1.position}")
print(f"Player 2 position: {player2.position}")

# Make players do something
player1.add_cmd("Forward")
player1.add_cmd("Look")
player2.add_cmd("Right")
player2.add_cmd("Forward")

# Run simulation for a few steps
for step in range(20):
    server.step()
    
    # Check for responses
    res1 = player1.get_res()
    res2 = player2.get_res()
    
    if res1:
        print(f"Player 1: {res1}")
    if res2:
        print(f"Player 2: {res2}")

print(f"Final positions - Player 1: {player1.position}, Player 2: {player2.position}")
```

Run it: `python3 simple_demo.py`

### 2. AI Training Environment

Now let's use the training environment:

```python
# training_demo.py
from environment import Category
import constant

# Create a survival challenge
env = Category["survival"](nb_players=3, food_amount=2, size=8)

# Reset environment
state = env.reset()

print(f"Number of players: {len(state['players'])}")
print("Starting simulation...")

# Simple AI: always look for food when hungry
for episode in range(5):
    state = env.reset()
    steps = 0
    
    print(f"\n--- Episode {episode + 1} ---")
    
    while steps < 50:  # Max 50 steps per episode
        # Get current state
        players = state["players"]
        
        # Simple strategy for each player
        for player_id, player in players.items():
            if not player.is_alive:
                continue
                
            food_level = player.inventory[constant.Resource.FOOD.value]
            
            if food_level <= 2:
                # Low food - try to find some
                player.add_cmd("Look")
                if steps % 3 == 0:  # Sometimes try to take food
                    player.add_cmd("Take food")
            else:
                # Safe - explore randomly
                import random
                action = random.choice(["Forward", "Left", "Right"])
                player.add_cmd(action)
        
        # Advance simulation
        state, rewards, done = env.step()
        steps += 1
        
        # Print rewards if any
        if rewards:
            for player_id, reward in rewards.items():
                if reward != 0:
                    print(f"  Player {player_id}: reward {reward:.2f}")
        
        # Check if episode ended
        if done:
            print(f"  Episode ended after {steps} steps")
            break
    
    # Episode summary
    alive_players = sum(1 for p in players.values() if p.is_alive)
    print(f"  Players alive: {alive_players}/{len(players)}")
```

Run it: `python3 training_demo.py`

### 3. Multi-Agent Cooperation

Let's try an environment that requires teamwork:

```python
# cooperation_demo.py
from environment import Category
from constant import Resource, ElevationRequirement

# Create group environment for level 2 incantation
env = Category["group"](level=1, nb_players=2, size=6)

state = env.reset()
print("Cooperation Challenge: Get players to work together for incantation")

# Check what's needed for level 1->2 incantation
required_players, required_resources = ElevationRequirement.requirements[1]
print(f"Need {required_players} players and resources: {required_resources}")

for episode in range(3):
    state = env.reset()
    print(f"\n--- Episode {episode + 1} ---")
    
    for step in range(100):
        players = list(state["players"].values())
        
        if not any(p.is_alive for p in players):
            print("All players died!")
            break
        
        # Strategy: gather at center and collect resources
        center = (3, 3)  # Center of 6x6 map
        
        for player in players:
            if not player.is_alive:
                continue
            
            x, y = player.position
            
            # Are we at the center?
            if (x, y) == center:
                # At center - look for resources or try incantation
                player.add_cmd("Look")
                
                # Try to take required resources
                for resource in required_resources:
                    if step % 10 == 0:  # Periodically try to take
                        player.add_cmd(f"Take {resource.value}")
                
                # Try incantation if we have resources
                if step % 20 == 0:
                    player.add_cmd("Incantation")
                    
            else:
                # Move toward center
                if x < center[0]:
                    player.add_cmd("Right")  # Face east
                    player.add_cmd("Forward")
                elif x > center[0]:
                    player.add_cmd("Left")   # Face west
                    player.add_cmd("Left")
                    player.add_cmd("Forward")
                elif y < center[1]:
                    player.add_cmd("Forward")  # Go south (default north)
                elif y > center[1]:
                    player.add_cmd("Right")    # Face south
                    player.add_cmd("Right")
                    player.add_cmd("Forward")
                
                # Broadcast position
                if step % 15 == 0:
                    player.add_cmd(f"Broadcast Moving to center from {x},{y}")
        
        # Step simulation
        state, rewards, done = env.step()
        
        # Check for successful incantation
        current_levels = [p.level for p in players if p.is_alive]
        if any(level > 1 for level in current_levels):
            print(f"SUCCESS! Players reached levels: {current_levels}")
            break
        
        if done:
            print(f"Episode ended at step {step}")
            break

print("\nTry modifying the strategy to improve cooperation!")
```

Run it: `python3 cooperation_demo.py`

## Next Steps

### For AI Researchers
1. **Read the [Training Guide](docs/TRAINING_GUIDE.md)** - Learn about RL integration, state representation, and training strategies
2. **Explore environments** - Try different environment types to understand various challenges
3. **Custom rewards** - Create custom reward functions for your specific research goals

### For Game Developers  
1. **Check the [API Reference](docs/API_REFERENCE.md)** - Understand all available commands and game mechanics
2. **Review [Architecture](docs/ARCHITECTURE.md)** - Learn how the system is designed
3. **Extend functionality** - Add new commands, resources, or game mechanics

### For Students
1. **Run the examples** - Start with simple scenarios and gradually increase complexity
2. **Read the [FAQ](docs/FAQ.md)** - Common questions and troubleshooting tips
3. **Experiment** - Try different strategies and see how they affect outcomes

## Common Patterns

### Debugging Your Agent
```python
# Add this to see what your agent is doing
def debug_player_state(player):
    print(f"Player {player.id}:")
    print(f"  Position: {player.position}")
    print(f"  Direction: {player.direction}")
    print(f"  Level: {player.level}")
    print(f"  Food: {player.inventory['food']}")
    print(f"  Commands queued: {len(player.command_queue)}")
    
    # Get latest response
    response = player.get_res()
    if response:
        print(f"  Last response: {response}")
```

### Simple Resource Collection
```python
def collect_resources(player):
    """Basic resource collection strategy."""
    # Look around
    player.add_cmd("Look")
    
    # Try to take valuable resources
    for resource in ["linemate", "deraumere", "sibur"]:
        player.add_cmd(f"Take {resource}")
    
    # Always ensure we have food
    if player.inventory["food"] < 5:
        player.add_cmd("Take food")
```

### Random Exploration
```python
import random

def explore_randomly(player):
    """Random movement for exploration."""
    action = random.choice([
        "Forward", "Forward", "Forward",  # Bias toward forward movement
        "Left", "Right"
    ])
    player.add_cmd(action)
```

That's it! You now have a working Zappy AI Training Environment. Explore the documentation and examples to build more sophisticated AI agents. Happy coding! 🚀
