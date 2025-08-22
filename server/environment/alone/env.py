from server.environment.environment import Environment
from server.environment.reward_constant import RewardConstants
import random
from server.logging_config import get_logger

logger = get_logger(__name__)

class EnvironmentAlone(Environment):
    """
    One agent alone, learns to move and collect resources. Ends when the player dies.
    Difficulty: ⭐
    """
    def __init__(self, level=1, size=5, seed=None, **kwargs):
        self.level = level
        super().__init__(size=size, nb_teams=1, nb_connect=1, seed=seed)
        logger.info("EnvironmentAlone initialized | level=%s size=%s", self.level, size)

    def _setup(self):
        self.add_player(0, self.level)
        logger.debug("EnvironmentAlone setup | players=%s", len(self.players))

    def _compute_reward(self, player, prev_state):
        reward = super()._compute_reward(player, prev_state)
        if player.level > prev_state["level"]:
            reward += RewardConstants.ALONE_LEVEL_BONUS
        return reward

    def _check_termination(self):
        if not self.players[0].is_alive:
            logger.info("EnvironmentAlone terminated: player died")
            return True
        if self.players[0].level >= 8:
            logger.info("EnvironmentAlone terminated: level reached >= 8")
            return True
        return False

    def shuffle_params(self):
        """Shuffle parameters for EnvironmentAlone"""
        self.level = random.randint(1, 7)
        super().shuffle_param(shuffle_size=True)
        logger.debug("EnvironmentAlone params shuffled | level=%s size=%s", self.level, self.size)
