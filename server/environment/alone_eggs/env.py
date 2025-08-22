from server.environment.environment import Environment
from server.environment.reward_constant import RewardConstants
from server import Player
import random
from server.logging_config import get_logger

logger = get_logger(__name__)


class EnvironmentAloneWithConnectSlot(Environment):
    """
    One agent with connect slot, learns to move, collect, and connect. Ends when the player dies.
    Difficulty: ⭐⭐
    """
    def __init__(self, level=1, size=5, nb_connect=5, seed=None, **kwargs):
        self.level = level
        super().__init__(size=size, nb_teams=1, nb_connect=nb_connect, seed=seed)
        logger.info("EnvironmentAloneWithConnectSlot initialized | level=%s size=%s nb_connect=%s", self.level, size, nb_connect)

    def _setup(self):
        self.add_player(0, self.level)
        logger.debug("EnvironmentAloneWithConnectSlot setup | players=%s", len(self.players))

    def _compute_reward(self, player: Player, prev_state):
        reward = super()._compute_reward(player, prev_state)
        if player.cur_cmd == player.fork:
            reward += RewardConstants.FORK_REWARD
        return reward

    def shuffle_params(self):
        """Shuffle parameters for EnvironmentAloneWithConnectSlot"""
        self.level = random.randint(1, 6)
        super().shuffle_param(shuffle_size=True, shuffle_nb_connect=True)
        logger.debug("EnvironmentAloneWithConnectSlot params shuffled | level=%s size=%s nb_connect=%s", self.level, self.size, getattr(self, 'nb_connect', self.init_nb_connect))
