# Configuration Guide

This guide explains how to configure and customize the Zappy AI Training Environment for different use cases and requirements.

## Environment Configuration

### Basic Environment Parameters

All environments accept these common parameters:

```python
from environment import Category

# Basic configuration
env = Category["alone"](
    size=10,           # Map dimensions (10x10)
    nb_teams=4,        # Number of teams
    nb_connect=5,      # Initial connection slots per team
    seed=42            # Random seed for reproducibility
)
```

### Environment-Specific Parameters

#### Alone Environment
```python
env = Category["alone"](
    level=3,           # Starting player level (1-7)
    size=15,           # Map size
    seed=123
)
```

#### Group Environment
```python
env = Category["group"](
    level=2,           # Starting level for all players
    nb_players=4,      # Number of players in the group
    size=12,           # Map size
    nb_teams=2,        # Number of teams
    nb_connect=3       # Connection slots per team
)
```

#### Survival Environment
```python
env = Category["survival"](
    nb_players=6,      # Number of players
    food_amount=3,     # Starting food per player
    size=8,            # Smaller map for scarcity
    seed=456
)
```

#### Incantation Environment
```python
env = Category["incantation"](
    level=4,           # Target incantation level
    size=10,           # Map size
    nb_teams=1         # Single team for focus
)
```

#### Random Group Environment
```python
env = Category["random_group"](
    nb_players=8,      # Total players
    size=20,           # Larger map
    level_range=(1, 5) # Random levels between 1-5
)
```

#### Close Incantation Environment
```python
env = Category["close_incantation"](
    level=3,           # Target level
    missing_resources=["sibur"],  # Which resources to omit
    size=8
)
```

## Resource Configuration

### Resource Density Settings

Modify global resource spawn rates:

```python
from constant import RESOURCE_DENSITY, Resource

# Backup original densities
original_densities = RESOURCE_DENSITY.copy()

# Create resource-rich environment
RESOURCE_DENSITY.update({
    Resource.FOOD: 0.8,        # More food
    Resource.LINEMATE: 0.5,    # More linemate
    Resource.DERAUMERE: 0.3,   # More deraumere
    # ... other resources
})

# Create environment
env = Category["survival"](size=10)

# Restore original densities
RESOURCE_DENSITY.update(original_densities)
```

### Custom Resource Distributions

```python
class ResourceRichEnvironment(Environment):
    def _setup(self):
        """Setup with custom resource distribution."""
        # Clear existing resources
        for y in range(self.map.size):
            for x in range(self.map.size):
                for resource in Resource:
                    self.map.tiles[y][x][resource] = 0
        
        # Add concentrated resource patches
        self._add_resource_patch(Resource.FOOD, center=(5, 5), radius=2, density=10)
        self._add_resource_patch(Resource.LINEMATE, center=(15, 15), radius=3, density=5)
    
    def _add_resource_patch(self, resource, center, radius, density):
        """Add concentrated resource area."""
        cx, cy = center
        for y in range(max(0, cy - radius), min(self.map.size, cy + radius + 1)):
            for x in range(max(0, cx - radius), min(self.map.size, cx + radius + 1)):
                distance = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
                if distance <= radius:
                    amount = max(1, int(density * (1 - distance / radius)))
                    self.map.tiles[y][x][resource] = amount
```

## Reward System Configuration

### Modifying Reward Constants

```python
from environment.reward_constant import RewardConstants

# Backup original values
original_rewards = {
    'DEATH_PENALTY': RewardConstants.DEATH_PENALTY,
    'LEVEL_PROGRESSION_BASE': RewardConstants.LEVEL_PROGRESSION_BASE,
    'FOOD_COLLECTION_BASE': RewardConstants.FOOD_COLLECTION_BASE
}

# Adjust for faster learning
RewardConstants.DEATH_PENALTY = -50        # Less harsh death penalty
RewardConstants.LEVEL_PROGRESSION_BASE = 20  # Higher level rewards
RewardConstants.FOOD_COLLECTION_BASE = 2.0   # More food incentive

# Use environment
env = Category["survival"](nb_players=4)

# Restore original values when done
for key, value in original_rewards.items():
    setattr(RewardConstants, key, value)
```

### Custom Reward Functions

```python
class CustomRewardEnvironment(Environment):
    def __init__(self, exploration_bonus=1.0, cooperation_bonus=2.0, **kwargs):
        super().__init__(**kwargs)
        self.exploration_bonus = exploration_bonus
        self.cooperation_bonus = cooperation_bonus
        self.visited_tiles = {}  # Track exploration
    
    def _compute_reward(self, player, prev_state):
        """Custom reward calculation."""
        reward = super()._compute_reward(player, prev_state)
        
        # Exploration bonus
        pos = player.position
        player_id = player.id
        if player_id not in self.visited_tiles:
            self.visited_tiles[player_id] = set()
        
        if pos not in self.visited_tiles[player_id]:
            self.visited_tiles[player_id].add(pos)
            reward += self.exploration_bonus
        
        # Cooperation bonus (nearby teammates)
        teammates_nearby = sum(1 for p in self.players 
                             if p.team == player.team 
                             and p != player 
                             and self._distance(p.position, pos) <= 2)
        reward += teammates_nearby * self.cooperation_bonus
        
        return reward
    
    def _distance(self, pos1, pos2):
        """Calculate toroidal distance between positions."""
        dx = min(abs(pos1[0] - pos2[0]), self.size - abs(pos1[0] - pos2[0]))
        dy = min(abs(pos1[1] - pos2[1]), self.size - abs(pos1[1] - pos2[1]))
        return (dx ** 2 + dy ** 2) ** 0.5
```

## Command Timing Configuration

### Adjusting Command Speeds

```python
from player import CommandTick, Player

# Backup original timings
original_timings = CommandTick.copy()

# Speed up commands for faster training
CommandTick.update({
    Player.forward: 3,      # Faster movement
    Player.look: 1,         # Instant observation
    Player.take: 3,         # Faster resource collection
    Player.set: 3,          # Faster resource dropping
    Player.incantation: 100 # Faster incantations
})

# Use environment
env = Category["incantation"](level=2)

# Restore original timings
CommandTick.update(original_timings)
```

### Custom Command Sets

```python
# Define custom command with timing
def special_move(self):
    """Move two tiles forward."""
    self.forward()
    self.forward()
    self.add_res("double_forward_ok")

# Add to player class
Player.special_move = special_move

# Register command
from player import Command, CommandTick
Command["SpecialMove"] = Player.special_move
CommandTick[Player.special_move] = 14  # Double forward time
```

## Map Configuration

### Custom Map Sizes and Shapes

```python
class LargeMapEnvironment(Environment):
    def __init__(self, **kwargs):
        kwargs['size'] = 50  # Override size
        super().__init__(**kwargs)
    
    def _setup(self):
        """Setup for large map training."""
        # Add landmarks or special areas
        self._create_resource_zones()
        self._create_safe_zones()
    
    def _create_resource_zones(self):
        """Create designated resource-rich areas."""
        zones = [
            ((10, 10), (15, 15)),  # Zone 1
            ((35, 35), (40, 40)),  # Zone 2
        ]
        
        for (x1, y1), (x2, y2) in zones:
            for y in range(y1, y2):
                for x in range(x1, x2):
                    # High resource density in zones
                    for resource in Resource:
                        if resource != Resource.FOOD:
                            self.map.tiles[y][x][resource] = 3
```

### Non-Square Maps

```python
class RectangularMapEnvironment(Environment):
    def __init__(self, width=20, height=10, **kwargs):
        self.width = width
        self.height = height
        kwargs['size'] = max(width, height)  # Use larger dimension
        super().__init__(**kwargs)
        
        # Override map creation
        self._create_rectangular_map()
    
    def _create_rectangular_map(self):
        """Create rectangular map instead of square."""
        # Clear existing tiles
        self.map.tiles = []
        
        # Create rectangular grid
        for y in range(self.height):
            row = []
            for x in range(self.width):
                tile = {resource: 0 for resource in Resource}
                row.append(tile)
            self.map.tiles.append(row)
        
        # Redistribute resources for new shape
        self.map._Map__map_resource()
```

## Team and Player Configuration

### Dynamic Team Management

```python
class DynamicTeamEnvironment(Environment):
    def __init__(self, max_players_per_team=5, **kwargs):
        super().__init__(**kwargs)
        self.max_players_per_team = max_players_per_team
    
    def add_player_dynamic(self, preferred_team=None):
        """Add player with dynamic team balancing."""
        if preferred_team and preferred_team in self.teams_name:
            team_counts = {team: sum(1 for p in self.players if p.team == team) 
                          for team in self.teams_name}
            
            if team_counts[preferred_team] < self.max_players_per_team:
                return self.server.add_player(preferred_team)
        
        # Find team with fewest players
        team_counts = {team: sum(1 for p in self.players if p.team == team) 
                      for team in self.teams_name}
        best_team = min(team_counts, key=team_counts.get)
        
        if team_counts[best_team] < self.max_players_per_team:
            return self.server.add_player(best_team)
        
        return None  # All teams full
```

### Player Skill Levels

```python
class SkillBasedEnvironment(Environment):
    def __init__(self, skill_distribution="normal", **kwargs):
        super().__init__(**kwargs)
        self.skill_distribution = skill_distribution
    
    def _setup(self):
        """Setup players with varied skill levels."""
        for player in self.players:
            skill_level = self._generate_skill_level()
            self._apply_skill_modifiers(player, skill_level)
    
    def _generate_skill_level(self):
        """Generate skill level based on distribution."""
        if self.skill_distribution == "normal":
            import random
            return max(1, min(7, int(random.gauss(3, 1))))
        elif self.skill_distribution == "uniform":
            import random
            return random.randint(1, 7)
        return 3  # Default
    
    def _apply_skill_modifiers(self, player, skill_level):
        """Apply skill-based modifications to player."""
        # Higher skill = more starting resources
        bonus_resources = skill_level - 1
        for resource in [Resource.LINEMATE, Resource.DERAUMERE]:
            player.inventory[resource.value] += bonus_resources
```

## Performance Configuration

### Optimization Settings

```python
class OptimizedEnvironment(Environment):
    def __init__(self, batch_commands=True, cache_lookups=True, **kwargs):
        super().__init__(**kwargs)
        self.batch_commands = batch_commands
        self.cache_lookups = cache_lookups
        self._position_cache = {}
        self._command_batch = []
    
    def step(self):
        """Optimized step with batching and caching."""
        if self.batch_commands:
            return self._step_with_batching()
        else:
            return super().step()
    
    def _step_with_batching(self):
        """Process multiple commands per step for speed."""
        # Collect all pending commands
        for player in self.players:
            if player.command_queue and not player.cur_cmd:
                self._command_batch.append((player, player.command_queue.pop(0)))
        
        # Execute batch
        for player, command in self._command_batch:
            # Process command
            pass
        
        self._command_batch.clear()
        return super().step()
```

### Memory Management

```python
class MemoryEfficientEnvironment(Environment):
    def __init__(self, cleanup_interval=1000, **kwargs):
        super().__init__(**kwargs)
        self.cleanup_interval = cleanup_interval
        self.step_count = 0
    
    def step(self):
        """Step with periodic memory cleanup."""
        result = super().step()
        
        self.step_count += 1
        if self.step_count % self.cleanup_interval == 0:
            self._cleanup_memory()
        
        return result
    
    def _cleanup_memory(self):
        """Periodic memory cleanup."""
        import gc
        
        # Clear dead player references
        self.players = [p for p in self.players if p.is_alive]
        
        # Clear response queues for efficiency
        for player in self.players:
            if len(player.response_queue) > 10:
                player.response_queue = player.response_queue[-5:]
        
        # Force garbage collection
        gc.collect()
```

## Configuration Files

### JSON Configuration

```json
{
  "environment": {
    "type": "survival",
    "size": 15,
    "nb_players": 6,
    "food_amount": 4,
    "seed": 12345
  },
  "rewards": {
    "death_penalty": -75,
    "level_progression": 15,
    "food_collection": 1.5,
    "exploration_bonus": 0.5
  },
  "commands": {
    "fast_mode": true,
    "forward_ticks": 5,
    "look_ticks": 1,
    "incantation_ticks": 200
  },
  "resources": {
    "food_density": 0.6,
    "linemate_density": 0.4,
    "rare_resource_multiplier": 2.0
  }
}
```

### Loading Configuration

```python
import json

def load_environment_from_config(config_file):
    """Load environment from JSON configuration."""
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Apply configuration
    env_config = config['environment']
    env_type = env_config.pop('type')
    
    # Apply reward configuration
    if 'rewards' in config:
        apply_reward_config(config['rewards'])
    
    # Apply command timing configuration
    if 'commands' in config:
        apply_command_config(config['commands'])
    
    # Apply resource configuration
    if 'resources' in config:
        apply_resource_config(config['resources'])
    
    # Create environment
    return Category[env_type](**env_config)

def apply_reward_config(reward_config):
    """Apply reward configuration."""
    for key, value in reward_config.items():
        attr_name = key.upper()
        if hasattr(RewardConstants, attr_name):
            setattr(RewardConstants, attr_name, value)

# Usage
env = load_environment_from_config('training_config.json')
```

This configuration guide provides comprehensive options for customizing the Zappy environment to meet specific training requirements and research objectives.
