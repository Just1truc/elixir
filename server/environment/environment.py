from server.environment.reward_constant import RewardConstants

import random
from server import constant
from server import Server
from server import Player
from server.logging_config import get_logger

logger = get_logger(__name__)


class Environment:
    """Base Environment class for all training environments"""
    
    def __init__(self, size: int = 10, nb_teams: int = 4, nb_connect: int = 5, seed: int = None):
        if seed:
            random.seed(seed)
        self.size = size
        self.init_nb_connect = nb_connect
        self.teams_name = [str(i) for i in range(nb_teams)]
        self.server = Server(self.size, self.init_nb_connect, self.teams_name)
        self.players: list[Player] = self.server.players
        self.map = self.server.map
        logger.info("Environment initialized | size=%s teams=%s nb_connect=%s", self.size, self.teams_name, self.init_nb_connect)
        self._setup()
        logger.debug("Environment setup complete | players=%s", len(self.players))

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
        for player, prev_state in zip(self.players, self.prev_states):
            rewards.append(self._compute_reward(player, prev_state))
        return rewards
    
    def step(self):
        """Execute 7 game tick == 1 action tick"""
        logger.debug("Environment step start | ticks=%s players=%s", self.server.ticks, len(self.players))
        self.prev_states = []
        for player in self.players:
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
        logger.debug("Environment step end | ticks=%s terminated=%s rewards_sum=%.2f", self.server.ticks, terminated, sum(rewards) if rewards else 0.0)
        return state, rewards, terminated

    def clone(self):
        """Create a deep copy of the current environment state"""
        logger.info("Cloning environment | size=%s teams=%s nb_connect=%s", self.size, self.teams_name, self.init_nb_connect)
        
        # Create a new environment instance (fresh state)
        cloned_env = self.__class__(
            size=self.size,
            nb_teams=len(self.teams_name),
            nb_connect=self.init_nb_connect,
            seed=None  # Explicitly avoid seed copying
        )
        
        # Replace with a deep-cloned server (includes players, map, teams)
        cloned_env.server = self.server.clone()
        cloned_env.map = cloned_env.server.map
        # Use the cloned server's players directly
        cloned_env.players = cloned_env.server.players
        logger.debug("Environment cloned | players=%s ticks=%s", len(cloned_env.players), cloned_env.server.ticks)
        
        return cloned_env

    def reset(self):
        """Create the Server to initiate the environment"""
        logger.info("Environment reset | nb_connect=%s", self.init_nb_connect)
        self.server.reset(self.init_nb_connect)
        self.map = self.server.map
        # Not useful to set players again, but done in case to avoid edgy case.
        self.players = self.server.players
        self._setup()
        logger.debug("Environment reset complete | players=%s", len(self.players))
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
            logger.warning("Requested state for unknown player_id=%s", player_id)
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
        logger.debug("Fetched player state | id=%s level=%s pos=%s", player_id, player.level, player.position)
        return player_state

    def add_player(self, team_idx: int, level: int = 1, position: tuple = None):
        """Add a player to the environment.
        Falls back to direct creation if no egg is available for the team."""
        team = self.teams_name[team_idx]
        p = self.server.add_player(team)
        if p is None:
            # Fallback: create player directly (use provided position or random)
            p = Player(len(self.server.players), self.server.map, team)
            if position is not None:
                p.position = position
            else:
                p.position = (random.randrange(0, self.size), random.randrange(0, self.size))
            self.server.players.append(p)
            logger.warning("No eggs available, created player directly | id=%s team=%s pos=%s", p.id, team, p.position)
        # Apply requested attributes
        p.level = level
        if position is not None:
            p.position = position
        logger.info("Player added | id=%s team=%s level=%s pos=%s", p.id, team, level, p.position)
        return p

    def shuffle_param(self, shuffle_size: bool = False, shuffle_nb_connect: bool = False, 
                     shuffle_nb_teams: bool = False, shuffle_seed: bool = False):
        """Shuffle environment parameters for randomization"""
        if shuffle_size:
            self.size = random.randrange(5, 42)
        if shuffle_nb_connect:
            self.nb_connect = random.randrange(1, 10)
            self.init_nb_connect = self.nb_connect
        if shuffle_nb_teams:
            new_nb_teams = random.randrange(1, 5)
            self.teams_name = [str(i) for i in range(new_nb_teams)]
        if shuffle_seed:
            new_seed = random.randint(1, 10000)
            random.seed(new_seed)
        logger.debug("Shuffled params | size=%s nb_connect=%s teams=%s", self.size, getattr(self, 'nb_connect', self.init_nb_connect), self.teams_name)

    def shuffle_all_params(self):
        """Shuffle all environment parameters"""
        self.shuffle_param(shuffle_size=True, shuffle_nb_connect=True, 
                          shuffle_nb_teams=True, shuffle_seed=True)
        logger.debug("Shuffled all params | size=%s nb_connect=%s teams=%s", self.size, getattr(self, 'nb_connect', self.init_nb_connect), self.teams_name)


