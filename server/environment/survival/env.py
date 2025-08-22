from server.environment.environment import Environment
from server.environment.reward_constant import RewardConstants
from server import constant
import random
from server.logging_config import get_logger

logger = get_logger(__name__)


class EnvironmentSurvival(Environment):
    """
    Players start with limited food. The map's food resources are reduced, forcing agents to manage scarcity and compete for survival. The game ends when all players die.
    Difficulty: ⭐⭐⭐⭐⭐
    """
    def __init__(self, nb_players=3, food_amount=3, size=8, seed=None, **kwargs):
        self.nb_players = nb_players
        self.food_amount = food_amount
        super().__init__(size=size, nb_teams=1, nb_connect=nb_players, seed=seed)
        logger.info("EnvironmentSurvival initialized | nb_players=%s food_amount=%s size=%s", nb_players, food_amount, size)

    def _setup(self):
        for i in range(self.nb_players):
            player = self.add_player(0)
            player.inventory[constant.Resource.FOOD.value] = self.food_amount
        self.map.total_resources[constant.Resource.FOOD] /= 5
        for x in range(self.size):
            for y in range(self.size):
                self.map.tiles[y][x][constant.Resource.FOOD.value] = max(
                    0,
                    self.map.tiles[y][x].get(
                        constant.Resource.FOOD.value, 0) - 2
                )
        logger.debug("EnvironmentSurvival setup | players=%s food_amount=%s", len(self.players), self.food_amount)

            
    def _compute_reward(self, player, prev_state):
        reward = super()._compute_reward(player, prev_state)
        food = player.inventory.get(constant.Resource.FOOD.value, 0)
        if food < 3 and prev_state["inventory"].get(constant.Resource.FOOD.value, 0) < food:
            reward += RewardConstants.CRITICAL_FOOD_REWARD
        return reward

    def _check_termination(self):
        terminated = all(not p.is_alive for p in self.players)
        if terminated:
            logger.info("EnvironmentSurvival terminated: all players dead")
        return terminated

    def shuffle_params(self):
        """Shuffle parameters for EnvironmentSurvival"""
        self.nb_players = random.randint(2, 6)
        self.food_amount = random.randint(1, 5)
        super().shuffle_param(shuffle_size=True)
        logger.debug("EnvironmentSurvival params shuffled | nb_players=%s food_amount=%s size=%s", self.nb_players, self.food_amount, self.size)
