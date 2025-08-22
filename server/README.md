# Zappy AI Training Environment

The Zappy AI Training Environment provides a socket-free simulation for developing and training AI agents to play the Zappy game. This environment allows you to focus on AI strategy development without dealing with network communication or real-time constraints.

## Features

- Complete simulation of Zappy game mechanics
- Multiple pre-configured training scenarios
- Direct access to game state for AI observation
- Socket-free implementation for faster training
- Customizable environment parameters
- Support for multi-agent training

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd zappy-python-server

# No additional dependencies required - uses Python standard library only

# Verify installation
python3 -c "from environment import Category; print('✓ Installation successful')"

# Run basic test
python3 environment/run_all_tests.py
```

## Environment Types

The training environment includes several pre-configured scenarios, each organized in its own module:

| Environment          | Description                                     | Parameters                     |
|----------------------|-------------------------------------------------|--------------------------------|
| `alone`              | Single player at specified level                | `level`, `size`               |
| `alone_eggs`         | Single player with connection slots available   | `level`, `size`               |
| `group`              | Multiple players at same level                  | `level`, `nb_players`, `size` |
| `random_group`       | Players at random levels (1-7)                 | `nb_players`, `size`          |
| `survival`           | Limited food survival challenge                | `nb_players`, `food_amount`, `size` |
| `incantation`        | Ready-to-perform elevation ritual              | `level`, `size`               |
| `close_incantation`  | Almost ready elevation ritual (missing resource) | `level`, `size`               |
| `strict_incantation` | Incantation with heavy movement penalties      | `level`, `size`               |

Each environment type is implemented in its own directory under `environment/` with dedicated test suites.

## Project Structure

```
zappy-python-server/
├── main.py                    # Entry point for running simulations
├── server.py                  # Core server simulation logic
├── player.py                  # Player class with all game commands
├── map.py                     # Game map and resource management
├── constant.py                # Game constants and enums
├── environment/               # Modular training environments
│   ├── __init__.py           # Package exports
│   ├── environment.py        # Base Environment class
│   ├── reward_constant.py    # Reward system constants
│   ├── run_all_tests.py      # Unified test runner
│   ├── alone/                # Solo player environment
│   │   ├── env.py           
│   │   └── test_alone.py    
│   ├── group/                # Multi-player environments
│   │   ├── env.py           
│   │   └── test_group.py    
│   ├── survival/             # Survival challenge environment
│   │   ├── env.py           
│   │   └── test_survival.py 
│   ├── incantation/          # Elevation ritual environments
│   │   ├── env.py           
│   │   └── test_incantation.py
│   └── ... (other environment types)
├── docs/                     # Comprehensive documentation
└── tests/                    # General test suites
```

## Quick Start

New to Zappy? Check out our **[Quick Start Guide](docs/QUICK_START.md)** for a step-by-step introduction with runnable examples.

## Testing

The project includes comprehensive test suites organized by environment type:

```bash
# Run all tests
python3 environment/run_all_tests.py

# Run tests for specific environment
python3 -m unittest environment.alone.test_alone
python3 -m unittest environment.group.test_group
python3 -m unittest environment.survival.test_survival
# ... etc for other environments
```

Each environment has its own test suite that validates:
- Environment initialization and setup
- Player behavior and state management
- Reward calculation accuracy
- Termination conditions
- Edge cases and error handling

## Documentation

Comprehensive documentation is available in the `docs/` directory:

- **[API Reference](docs/API_REFERENCE.md)** - Complete API documentation for all classes and methods
- **[Training Guide](docs/TRAINING_GUIDE.md)** - Comprehensive guide for training AI agents
- **[Architecture Overview](docs/ARCHITECTURE.md)** - System design and component interactions
- **[Configuration Guide](docs/CONFIGURATION.md)** - Environment and parameter customization
- **[Development Guide](docs/DEVELOPMENT.md)** - Contributing and development setup
- **[FAQ](docs/FAQ.md)** - Frequently asked questions and troubleshooting
- **[Reward System](docs/reward.md)** - Detailed reward system documentation

## Basic Usage

### Server Simulation

```python
import server

# Create server with 42x42 map, 5 connections, and two teams
main = server.Server(42, 5, ["red", "blue"])

# Add player to red team
player = main.add_player("red")

# Run simulation
for i in range(80):
    # Queue commands
    player.add_cmd(player.forward)
    
    # Advance simulation
    main.step()
    
    # Get responses
    res = player.get_res()
    if res:
        print(res)
    
    # Check player status
    if not player.is_alive:
        print("Player died!")
        break
```

### Training Environment

```python
from environment import Category
import constant

# Create survival environment
env = Category["survival"](nb_players=4, food_amount=2, size=10)

# Or try other environment types:
# env = Category["alone_eggs"](level=3, size=15)          # Solo with connections
# env = Category["strict_incantation"](level=2, size=8)   # Incantation with penalties
# env = Category["group"](level=2, nb_players=3, size=12) # Group training

# Reset environment to initial state
state = env.reset()

# Run simulation
for step in range(100):
    state, rewards, terminated = env.step()
    
    # AI decision-making would happen here
    for i, player in enumerate(state["players"]):
        if player.inventory[constant.Resource.FOOD.value] < 3:
            player.add_cmd(player.take, constant.Resource.FOOD.value)
        else:
            player.add_cmd(player.forward)
    
    # Check termination condition
    if terminated:
        print(f"Environment terminated at step {step}")
        print(f"Final rewards: {rewards}")
        break
```

## Key Classes

### Player Class
```python
class Player:
    def forward(self): ...
    def right(self): ...
    def left(self): ...
    def look(self): ...
    def inventory_cmd(self): ...
    def broadcast(self, text): ...
    def connect_nbr(self): ...
    def fork(self): ...
    def eject(self): ...
    def take(self, resource): ...
    def set(self, resource): ...
    def incantation(self): ...
    
    # Command management
    def add_cmd(self, func): ...
    def get_res(self): ...
    
    # Properties
    position
    direction
    level
    inventory
    is_alive
```

### Environment Class
```python
class Environment:
    def __init__(self, size=10, nb_teams=4, nb_connect=5, seed=None): ...
    def reset(self): ...
    def step(self): ...  # Returns (state, rewards, terminated)
    def get_state(self): ...
    def get_player_state(self, player_id): ...
    def add_player(self, team_idx, level=1, position=None): ...
    def clone(self): ...  # Create deep copy of current state
    def shuffle_param(self, shuffle_size, shuffle_nb_connect): ...
    
    # Protected methods for subclassing
    def _setup(self): ...  # Override for custom initialization
    def _compute_reward(self, player, prev_state): ...  # Override for custom rewards
    def _check_termination(self): ...  # Override for custom termination
```

## State Observation

The environment state includes:
```python
{
    "tiles": 2D array of map tiles with resources,
    "incanted_tiles": tiles currently in elevation ritual,
    "total_resources": dictionary of total resources on map,
    "players": list of Player objects
}
```

## Training Your AI

### Observation Space
- Player position and orientation
- Player inventory (food, resources)
- Nearby tiles (from `look()` command)
- Available connections (`connect_nbr`)
- Current level

### Action Space
- Movement: `forward`, `left`, `right`
- Resource management: `take`, `set`
- Communication: `broadcast`
- Special actions: `fork`, `eject`, `incantation`
- Information gathering: `look`, `inventory`, `connect_nbr`

### Reward Design

The environment provides detailed reward feedback for AI training:

#### Base Rewards (all environments)
- **Step penalty**: Small negative reward each step to encourage efficiency
- **Level progression**: Large positive reward for successful elevation
- **Food collection**: Positive reward for gathering food
- **Starvation penalty**: Negative reward when food is critically low
- **Death penalty**: Large negative reward for player death

#### Environment-Specific Rewards
- **Survival**: Additional penalties for low food reserves
- **Incantation**: Bonuses for successful ritual completion
- **Strict Incantation**: Heavy penalties for movement during rituals
- **Group**: Coordination bonuses for team actions

The reward system is fully customizable through `environment/reward_constant.py`.

## Creating Custom Environments

To create a new training environment:

1. **Create environment directory**: `environment/my_custom_env/`
2. **Implement environment class**: Create `env.py` with your environment logic
3. **Add test suite**: Create `test_my_custom_env.py` with comprehensive tests
4. **Register environment**: Add to `Category` mapping in `environment/environment.py`

```python
# environment/my_custom_env/env.py
from environment.environment import Environment

class EnvironmentMyCustom(Environment):
    def _setup(self):
        """Custom environment initialization"""
        # Add your setup logic here
        pass
    
    def _compute_reward(self, player, prev_state):
        """Custom reward calculation"""
        base_reward = super()._compute_reward(player, prev_state)
        # Add your reward logic here
        return base_reward
    
    def _check_termination(self):
        """Custom termination conditions"""
        # Add your termination logic here
        return False
```

See `environment/template/my_env.py` for a complete template.

## Development

### Prerequisites
- Python 3.8+
- Standard library modules (no external dependencies)

### Running the Project
```bash
# Basic simulation
python3 main.py

# Environment testing
python3 environment/main.py

# Run specific environment
python3 -c "from environment import Category; env = Category['survival'](size=10); print(env.reset())"
```

### Contributing
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes and add tests
4. Run all tests: `python3 environment/run_all_tests.py`
5. Commit your changes: `git commit -am 'Add new feature'`
6. Push to the branch: `git push origin feature/my-feature`
7. Submit a pull request

### Code Style
- Follow PEP 8 style guidelines
- Use type hints for all function parameters and return values
- Include comprehensive docstrings for all classes and methods
- Maintain test coverage for new features

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.