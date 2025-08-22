# Development Guide

This guide provides comprehensive information for developers working on the Zappy AI Training Environment, including setup, contribution guidelines, and development best practices.

## Development Setup

### Prerequisites
- Python 3.9 or higher
- Git for version control
- Optional: Virtual environment tool (venv, conda)

### Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd zappy-python-server

# Create virtual environment (recommended)
python3 -m venv zappy-env
source zappy-env/bin/activate  # On Windows: zappy-env\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests to verify setup
python3 tests/tests.py
```

### Project Structure

```
zappy-python-server/
├── main.py                 # Entry point for basic simulation
├── server.py              # Core game server logic
├── player.py              # Player class and commands
├── map.py                 # Game map and resources
├── constant.py            # Game constants and enums
├── environment/           # AI training environments
│   ├── environment.py     # Base environment class
│   ├── main.py           # Environment examples
│   ├── reward_constant.py # Reward system constants
│   ├── tests.py          # Environment-specific tests
│   └── template/         # Environment templates
│       └── my_env.py     # Custom environment template
├── tests/                # Test suite
│   └── tests.py         # Comprehensive test cases
├── docs/                 # Documentation
│   ├── API_REFERENCE.md  # Complete API documentation
│   ├── TRAINING_GUIDE.md # AI training guide
│   ├── ARCHITECTURE.md   # System architecture
│   └── reward.md        # Reward system details
└── README.md            # Project overview
```

## Development Workflow

### Code Organization Principles

1. **Single Responsibility**: Each class/module has one clear purpose
2. **Loose Coupling**: Minimal dependencies between components
3. **High Cohesion**: Related functionality grouped together
4. **Testability**: All components easily unit tested

### Adding New Features

#### 1. New Player Commands

```python
# In player.py
def new_command(self, param: str = None):
    """
    Description of the new command.
    Args:
        param: Optional parameter description
    """
    # Implementation
    self.add_res("ok")  # or appropriate response

# Update command mappings
Command["NewCommand"] = Player.new_command
CommandTick[Player.new_command] = 10  # Command duration in ticks
```

#### 2. New Environment Types

```python
# In environment/environment.py or separate file
class NewEnvironment(Environment):
    def __init__(self, custom_param=5, **kwargs):
        super().__init__(**kwargs)
        self.custom_param = custom_param
    
    def _setup(self):
        """Initialize environment-specific conditions"""
        # Custom initialization logic
        pass
    
    def _compute_reward(self, player, prev_state):
        """Calculate environment-specific rewards"""
        reward = super()._compute_reward(player, prev_state)
        # Add custom reward logic
        return reward
    
    def _check_termination(self):
        """Define custom termination conditions"""
        # Return True when episode should end
        return False

# Register in Category dictionary
from environment import Category
Category["new_env"] = NewEnvironment
```

#### 3. New Resource Types

```python
# In constant.py
class Resource(Enum):
    # Existing resources...
    NEW_RESOURCE = "new_resource"

# Update mappings
RESOURCE_ENUM["new_resource"] = Resource.NEW_RESOURCE
RESOURCE_DENSITY[Resource.NEW_RESOURCE] = 0.05

# Update elevation requirements if needed
ElevationRequirement.requirements[8] = (8, {
    Resource.NEW_RESOURCE: 5,
    # Other requirements...
})
```

### Testing Guidelines

#### Unit Testing Standards

```python
import unittest

class TestNewFeature(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.server = Server(size=5, nb_eggs=1, teams=["test"])
        self.player = self.server.add_player("test")
    
    def test_feature_success_case(self):
        """Test successful execution of new feature."""
        # Arrange
        initial_state = self.player.inventory.copy()
        
        # Act
        self.player.new_command()
        
        # Assert
        self.assertIn("ok", self.player.response_queue[-1])
    
    def test_feature_failure_case(self):
        """Test failure conditions of new feature."""
        # Test edge cases and error conditions
        pass
    
    def tearDown(self):
        """Clean up after each test method."""
        pass
```

#### Integration Testing

```python
class TestEnvironmentIntegration(unittest.TestCase):
    def test_full_simulation_cycle(self):
        """Test complete environment lifecycle."""
        env = Category["test_env"](size=5)
        
        # Test reset
        initial_state = env.reset()
        self.assertIsInstance(initial_state, dict)
        
        # Test multiple steps
        for _ in range(10):
            state, rewards, done = env.step()
            if done:
                break
        
        # Verify final state consistency
        self.assertTrue(self._validate_state(state))
```

### Code Quality Standards

#### Style Guidelines

Follow PEP 8 with these specific conventions:

```python
# Class names: PascalCase
class PlayerManager:
    pass

# Function/variable names: snake_case
def calculate_reward(player_state):
    total_reward = 0
    return total_reward

# Constants: UPPER_SNAKE_CASE
MAX_PLAYERS = 100
RESOURCE_SPAWN_RATE = 0.5

# Type hints for all public methods
def process_command(self, cmd: str, params: list[str] = None) -> bool:
    pass
```

#### Documentation Standards

```python
def complex_function(param1: int, param2: str = "default") -> tuple[int, str]:
    """
    Brief description of function purpose.
    
    Longer description if needed, explaining the logic,
    assumptions, or important behaviors.
    
    Args:
        param1: Description of first parameter
        param2: Description with default value mention
    
    Returns:
        Tuple containing (result_value, status_message)
    
    Raises:
        ValueError: When param1 is negative
        TypeError: When param2 is not string
    
    Example:
        >>> result, status = complex_function(5, "test")
        >>> print(f"Result: {result}, Status: {status}")
        Result: 25, Status: processed
    """
    if param1 < 0:
        raise ValueError("param1 must be non-negative")
    
    # Implementation
    return param1 * 5, f"{param2} processed"
```

#### Error Handling Best Practices

```python
class GameError(Exception):
    """Base exception for game-related errors."""
    pass

class InvalidCommandError(GameError):
    """Raised when an invalid command is executed."""
    pass

def safe_command_execution(player, command):
    """Execute command with proper error handling."""
    try:
        result = command(player)
        return result
    except InvalidCommandError as e:
        logger.warning(f"Invalid command for player {player.id}: {e}")
        player.add_res("ko")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in command execution: {e}")
        # Graceful degradation
        return None
```

## Performance Optimization

### Profiling and Benchmarking

```python
import cProfile
import time

def profile_environment_performance():
    """Profile environment step performance."""
    env = Category["survival"](nb_players=10, size=20)
    
    # Profile setup
    pr = cProfile.Profile()
    pr.enable()
    
    # Run simulation
    env.reset()
    start_time = time.time()
    
    for _ in range(1000):
        state, rewards, done = env.step()
        if done:
            env.reset()
    
    end_time = time.time()
    pr.disable()
    
    # Results
    print(f"Steps per second: {1000 / (end_time - start_time):.2f}")
    pr.print_stats(sort='cumulative')
```

### Memory Optimization

```python
import gc
import psutil
import os

def monitor_memory_usage():
    """Monitor memory usage during simulation."""
    process = psutil.Process(os.getpid())
    
    def get_memory_mb():
        return process.memory_info().rss / 1024 / 1024
    
    initial_memory = get_memory_mb()
    env = Category["group"](nb_players=100, size=50)
    
    # Monitor memory growth
    for episode in range(100):
        env.reset()
        for step in range(1000):
            env.step()
        
        if episode % 10 == 0:
            current_memory = get_memory_mb()
            print(f"Episode {episode}: {current_memory:.1f} MB "
                  f"(+{current_memory - initial_memory:.1f})")
            
            # Force garbage collection
            gc.collect()
```

## Debugging and Troubleshooting

### Logging Configuration

```python
import logging

# Configure logging for development
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('zappy_debug.log'),
        logging.StreamHandler()
    ]
)

# Use in code
logger = logging.getLogger(__name__)

def debug_player_action(player, action):
    logger.debug(f"Player {player.id} executing {action} at {player.position}")
```

### Debug Utilities

```python
def visualize_game_state(server):
    """Print ASCII representation of current game state."""
    print(f"Tick: {server.ticks}")
    print("Map (P=Player, R=Resource, E=Egg):")
    
    # Create visual grid
    grid = [['.' for _ in range(server.map.size)] 
            for _ in range(server.map.size)]
    
    # Mark players
    for player in server.players:
        if player.is_alive:
            x, y = player.position
            grid[y][x] = f'P{player.id}'
    
    # Mark resources (simplified)
    for y in range(server.map.size):
        for x in range(server.map.size):
            if grid[y][x] == '.' and any(count > 0 for count in server.map.tiles[y][x].values()):
                grid[y][x] = 'R'
    
    # Mark eggs
    for team, eggs in server.map.eggs_tile.items():
        for x, y in eggs:
            if grid[y][x] == '.':
                grid[y][x] = 'E'
    
    # Print grid
    for row in grid:
        print(' '.join(f'{cell:>3}' for cell in row))
```

### Common Issues and Solutions

#### Issue: Slow Simulation Performance
```python
# Solution: Optimize hot paths
def optimized_player_lookup(self, position):
    """Cache player positions for faster lookup."""
    if not hasattr(self, '_position_cache'):
        self._position_cache = {}
    
    # Rebuild cache if stale
    if self._cache_tick != self.server.ticks:
        self._position_cache.clear()
        for player in self.server.players:
            if player.is_alive:
                pos = player.position
                if pos not in self._position_cache:
                    self._position_cache[pos] = []
                self._position_cache[pos].append(player)
        self._cache_tick = self.server.ticks
    
    return self._position_cache.get(position, [])
```

#### Issue: Memory Leaks
```python
# Solution: Proper cleanup in environment reset
def reset_with_cleanup(self):
    """Reset environment with proper memory cleanup."""
    # Clear circular references
    for player in self.players:
        player.map = None
        player.command_queue.clear()
        player.response_queue.clear()
    
    # Clear map references
    self.map.players.clear()
    self.map.incanted_tile.clear()
    
    # Recreate clean state
    self.__init__(self.size, self.nb_teams, self.init_nb_connect)
```

## Contribution Guidelines

### Pull Request Process

1. **Fork and Branch**: Create feature branch from main
2. **Implement**: Add feature with tests and documentation
3. **Test**: Ensure all tests pass
4. **Document**: Update relevant documentation
5. **Review**: Submit PR with clear description

### Commit Message Format

```
type(scope): brief description

Detailed explanation of changes, including:
- What was changed and why
- Any breaking changes
- Related issue numbers

Examples:
feat(player): add new broadcast command with direction filtering
fix(environment): resolve memory leak in state reset
docs(api): update player command documentation
test(server): add integration tests for player management
```

### Code Review Checklist

- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] Documentation updated
- [ ] No performance regressions
- [ ] Backward compatibility maintained
- [ ] Error handling implemented
- [ ] Edge cases considered

## Release Process

### Version Management
Follow semantic versioning (MAJOR.MINOR.PATCH):
- MAJOR: Breaking API changes
- MINOR: New features, backward compatible
- PATCH: Bug fixes, backward compatible

### Pre-release Checklist
- [ ] All tests pass
- [ ] Performance benchmarks acceptable
- [ ] Documentation complete
- [ ] Breaking changes documented
- [ ] Migration guide provided (if needed)

This development guide ensures consistent, high-quality contributions to the Zappy AI Training Environment project.
