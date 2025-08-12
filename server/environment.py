
""" Global Imports """
import random

""" Local Imports """
import server.constant as constant
from server.server import Server
from server.player import Player
from server.constant import RewardConstants

class Environment:
    # Default value for training -- Can be randomized
    def __init__(self, size: int = 10, nb_teams: int = 4, nb_connect: int = 5, seed: int = None):
        if seed:
            random.seed(seed)
        self.size = size
        self.init_nb_connect = nb_connect
        self.map = None
        self.teams_name = [str(i) for i in range(nb_teams)]
        self.server = Server(self.size, self.init_nb_connect, self.teams_name)
        self.players: list[Player] = self.server.players
        self.map = self.server.map
        self._setup()

    def _setup(self):
        """To be overridden by subclasses"""
        pass

    def _compute_reward(self, player: Player, prev_state):
        """Base reward function - common across all environments"""
        reward = 0
        
        # Step penalty to encourage efficiency
        reward += RewardConstants.STEP_PENALTY_FACTOR * self.server.ticks / 7
        
        # Death penalty
        if not player.is_alive and prev_state["alive"]:
            reward += RewardConstants.DEATH_PENALTY
            
        # Level progression reward
        if player.level > prev_state["level"]:
            reward += RewardConstants.LEVEL_PROGRESSION_BASE * (
                player.level - prev_state["level"])
            
        # Food collection reward
        food_diff = player.inventory.get(constant.Resource.FOOD.value, 0) - prev_state["inventory"].get(constant.Resource.FOOD.value, 0)
        if food_diff > 0:
            reward += food_diff * RewardConstants.FOOD_COLLECTION_BASE
        
        # Starvation penalty
        food = player.inventory.get(constant.Resource.FOOD.value, 0)
        if food < 2:
            reward += RewardConstants.STARVATION_PENALTY
            
        return reward


    def _check_termination(self):
        """Base termination condition - never terminates by default"""
        return False

    def reward(self):
        """Calculate rewards for all players based on state changes"""
        rewards = []
        for player, prev_state in zip(self.players.values(), self.prev_states):
            rewards.append(self._compute_reward(player, prev_state))
        return rewards
    
    # observation, reward, terminated, truncated, info
    def step(self) -> tuple[dict, list, bool]:
        """Execute 7 game tick == 1 action tick"""
        self.prev_states = []
        for player in self.players.values():
            self.prev_states.append({
                "id": player.id,
                "level": player.level,
                "inventory": player.inventory.copy(),
                "alive": player.is_alive
            })
        
        for _ in range(7):
            self.server.step()
        state = self.get_state()
        rewards = self.reward() 
        terminated = self._check_termination()
        return state, rewards, terminated

    def clone(self):
        """Create a deep copy of the current environment state"""
        
        # Create a new environment instance
        cloned_env = self.__class__(
            size=self.size,
            nb_teams=len(self.teams_name),
            nb_connect=self.init_nb_connect,
            seed=None  # Explicitly avoid seed copying
        )
        
        # Manually copy the server state
        cloned_env.server = self.server.clone()
        cloned_env.map = cloned_env.server.map
        
        # Recreate players with their current state
        cloned_env.players = []
        for original_player in self.players:
            # Find which team this player belongs to
            team_idx = self.teams_name.index(original_player.team)
            
            # Create new player in cloned environment
            cloned_player = cloned_env.add_player(
                team_idx=team_idx,
                level=original_player.level,
                position=original_player.position
            )
            
            # Copy player attributes
            cloned_player.direction = original_player.direction
            cloned_player.inventory = original_player.inventory.copy()
            cloned_player.is_alive = original_player.is_alive
            cloned_player.command_queue = original_player.command_queue.copy()
            cloned_player.response_queue = original_player.response_queue.copy()
        
        return cloned_env

    def reset(self):
        """Create the Server to initiate the environment"""
        self.players.clear()
        self.server.reset(self.init_nb_connect)
        self.map = self.server.map
        self._setup()
        return self.get_state()

    def get_state(self):
        """Return current environment state for AI"""
        return {
            "tiles": self.map.tiles,
            "incanted_tiles": self.map.incanted_tile,
            "total_resources": self.map.total_resources,
            "players": self.players
        }

    def get_player_state(self, player_id):
        """
        Get the state from a specific player's perspective
        Includes player attributes and inventory information
        """
        if player_id not in self.players:
            return None
            
        player = self.players[player_id]
        
        # Build player state dictionary
        player_state = {
            "id": player_id,
            "position": player.position,
            "direction": player.direction,
            "level": player.level,
            "inventory": player.inventory.copy(),
            "is_alive": player.is_alive,
            "action_queue_size": len(player.command_queue),
            "response_queue_size": len(player.response_queue),
        }
        return player_state


    # Necessary since team's name are pre-defined
    def add_player(self, team_idx: int, level: int = 1, position: tuple = None):
        team = self.teams_name[team_idx]
        p = self.server.add_player(team)
        if p == None:
            return None
        p.level = level
        if position:
            p.position = position
        return p

    def shuffle_param(self, shuffle_size: bool, shuffle_nb_connect: bool):
        if shuffle_size:
            self.size = random.randrange(5, 42)
        if shuffle_nb_connect:
            self.nb_connect = random.randrange(1, 5)
            self.init_nb_connect = self.nb_connect


class EnvironmentAlone(Environment):
    """Single player environment"""

    def __init__(self, level=1, size=5, **kwargs):
        self.level = level
        super().__init__(size=size, nb_teams=1, nb_connect=1, **kwargs)

    def _setup(self):
        self.add_player(0, self.level)

    def _compute_reward(self, player, prev_state):
        """Reward for single player environment"""
        reward = super()._compute_reward(player, prev_state)
        
        # Additional rewards for alone environment
        if player.level > prev_state["level"]:
            reward += RewardConstants.ALONE_LEVEL_BONUS
            
        return reward
    
    def _check_termination(self):
        """Terminate when player dies or reaches max level"""
        if not self.players[0].is_alive:
            return True
        if self.players[0].level >= 8:
            return True
        return False


class EnvironmentAloneWithConnectSlot(Environment):
    """Single player environment with many eggs"""

    def __init__(self, level=1, size=5, **kwargs):
        self.level = level
        super().__init__(size=size, nb_teams=1, nb_connect=10, **kwargs)

    def _setup(self):
        self.add_player(0, self.level)

    def _compute_reward(self, player: Player, prev_state):
        """Reward for single player with many eggs"""
        reward = super()._compute_reward(player, prev_state)
        
        # Reward for using fork command
        if player.cur_cmd == player.fork:
            reward += RewardConstants.FORK_REWARD
            
        return reward


class EnvironmentGroup(Environment):
    """Group of players at same level"""

    def __init__(self, level, nb_players, size=10, **kwargs):
        self.level = level
        self.nb_players = nb_players
        super().__init__(size=size, nb_teams=1, nb_connect=nb_players, **kwargs)

    def _setup(self):
        # Add players clustered around center
        center = self.size // 2
        for _ in range(self.nb_players):
            x = center + random.randint(-1, 1)
            y = center + random.randint(-1, 1)
            self.add_player(0, self.level, (x, y))
    
    def _compute_reward(self, player, prev_state):
        """Reward for group environment"""
        reward = super()._compute_reward(player, prev_state)
        
        # Reward for cooperative behavior
        if player.last_command == player.broadcast:
            reward += RewardConstants.BROADCAST_REWARD
            
        # Reward for being near teammates
        teammate_near = any(
            p != player and abs(p.position[0]-player.position[0]) <= 1 
            and abs(p.position[1]-player.position[1]) <= 1 
            for p in self.players
        )
        if teammate_near:
            reward += RewardConstants.TEAMMATE_PROXIMITY_REWARD
            
        return reward
    
    def _check_termination(self):
        """Terminate when any player reaches target level"""
        for player in self.players:
            if player.level >= self.level + 1:  # Target: reach next level
                return True
        return False

class EnvironmentRandomGroup(Environment):
    """Group of players at random levels"""

    def __init__(self, nb_players, size=15, **kwargs):
        self.nb_players = nb_players
        super().__init__(size=size, nb_teams=1, nb_connect=nb_players, **kwargs)

    def _setup(self):
        # Add players at random positions and levels
        for i in range(self.nb_players):
            level = random.randint(1, 7)
            self.add_player(0, level)
    
    def _compute_reward(self, player, prev_state):
        """Reward for random group environment"""
        reward = super()._compute_reward(player, prev_state)
        
        # Extra reward for higher level players helping lower levels
        if player.level > 1 and player.cur_cmd in [player.set, player.broadcast]:
            reward += RewardConstants.HELPING_REWARD
            
        return reward


class EnvironmentSurvival(Environment):
    """Survival scenario with low food"""

    def __init__(self, nb_players=3, food_amount=3, size=8, **kwargs):
        self.nb_players = nb_players
        self.food_amount = food_amount
        super().__init__(size=size, nb_teams=1, nb_connect=nb_players, **kwargs)

    def _setup(self):
        # Add players with low food
        for i in range(self.nb_players):
            player = self.add_player(0)
            player.inventory[constant.Resource.FOOD.value] = self.food_amount

        # Reduce food on map
        for x in range(self.size):
            for y in range(self.size):
                self.map.tiles[y][x][constant.Resource.FOOD.value] = max(
                    0,
                    self.map.tiles[y][x].get(
                        constant.Resource.FOOD.value, 0) - 2
                )
    
    def _compute_reward(self, player, prev_state):
        """Reward for survival environment"""
        reward = super()._compute_reward(player, prev_state)
        
        # Big reward for finding food when low
        food = player.inventory.get(constant.Resource.FOOD.value, 0)
        if food < 3 and prev_state["inventory"].get(constant.Resource.FOOD.value, 0) < food:
            reward += RewardConstants.CRITICAL_FOOD_REWARD
            
        return reward
    
    def _check_termination(self):
        """Terminate when all players die"""
        return all(not p.is_alive for p in self.players.values())



class EnvironmentIncantation(Environment):
    """Ready-to-incant environment"""

    def __init__(self, level, size=10, **kwargs):
        self.level = level
        self.nb_players = constant.ElevationRequirement.requirements[level][0]
        super().__init__(size=size, nb_teams=1, **kwargs)

    def _setup(self):
        center = self.size // 2
        tile = self.map.tiles[center][center]

        # Add required resources to center tile
        _, resources = constant.ElevationRequirement.requirements[self.level]
        for resource, count in resources.items():
            tile[resource.value] = count

        # Add players at center
        for i in range(self.nb_players):
            self.add_player(0, self.level, (center, center))

    def _compute_reward(self, player, prev_state):
        """Reward for incantation environment"""
        reward = super()._compute_reward(player, prev_state)
        
        # Big reward for starting incantation
        if player.cur_cmd == player.incantation:
            reward += RewardConstants.INCANTATION_START_REWARD
            
        # Extra reward for having required resources
        _, resources_req = constant.ElevationRequirement.requirements[self.level]
        has_resources = all(
            self.map.tiles[player.position[1]][player.position[0]].get(res.value, 0) >= count
            for res, count in resources_req.items()
        )
        if has_resources:
            reward += RewardConstants.RESOURCE_READINESS_REWARD
            
        return reward
    
    def _check_termination(self):
        """Terminate when incantation completes or fails"""
        center = self.size // 2
        tile = self.map.tiles[center][center]
        
        # Check if incantation started
        if any(p.cur_cmd == p.incantation for p in self.players):
            return True
            
        # Check if requirements are no longer met
        _, resources_req = constant.ElevationRequirement.requirements[self.level]
        missing_resource = any(
            tile.get(res.value, 0) < count
            for res, count in resources_req.items()
        )
        if missing_resource:
            return True
            
        return False


class EnvironmentCloseToIncantation(EnvironmentIncantation):
    """Almost ready-to-incant environment"""

    def __init__(self, level, size=10, **kwargs):
        super().__init__(size=size, level=level, **kwargs)

    def _setup(self):
        center = self.size // 2
        tile = self.map.tiles[center][center]

        # Add required resources to center tile
        _, resources = constant.ElevationRequirement.requirements[self.level]
        for resource, count in resources.items():
            # Leave one resource missing
            tile[resource.value] = max(0, count - 1)

        # Place missing resources nearby
        for resource, count in resources.items():
            if count > 0:
                x = (center + 1) % self.size
                y = center
                self.map.tiles[y][x][resource.value] = 1

        # Add players at center
        for i in range(self.nb_players):
            self.add_player(0, self.level, (center, center))

    def _compute_reward(self, player, prev_state):
        """Reward for close-to-incantation environment"""
        reward = super()._compute_reward(player, prev_state)
        
        # Reward for collecting the missing resource
        _, resources_req = constant.ElevationRequirement.requirements[self.level]
        for resource, count in resources_req.items():
            prev_count = prev_state["inventory"].get(resource.value, 0)
            current_count = player.inventory.get(resource.value, 0)
            if current_count > prev_count:
                reward += RewardConstants.RESOURCE_COLLECTION_REWARD
                
        # Reward for bringing resource to center
        center = self.size // 2
        if player.position == (center, center) and player.cur_cmd == player.set:
            reward += RewardConstants.RESOURCE_DELIVERY_REWARD
            
        return reward

# Simple
# Alone lvl:x
# Group lvl:x nb:n
# Group lvl:[1-7] nb:n

# Survival (<4 food)
# Random distribution

# Incantation
# Can incante - lvl:x
# Close to incante - lvl:x

# Environment category mapping
Category = {
    "alone": EnvironmentAlone,
    "alone_eggs": EnvironmentAloneWithConnectSlot,
    "group": EnvironmentGroup,
    "random_group": EnvironmentRandomGroup,
    "survival": EnvironmentSurvival,
    "incantation": EnvironmentIncantation,
    "close_incantation": EnvironmentCloseToIncantation,
}
