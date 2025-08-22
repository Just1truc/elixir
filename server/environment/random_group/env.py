from server.environment.environment import Environment
from server.environment.reward_constant import RewardConstants
import random
from server.logging_config import get_logger

logger = get_logger(__name__)


class EnvironmentRandomGroup(Environment):
    """
    Agents grouped randomly, must adapt and cooperate. Ends when all players die.
    Difficulty: ⭐⭐⭐⭐
    """
    def __init__(self, nb_players, size=15, seed=None, **kwargs):
        self.nb_players = nb_players
        super().__init__(size=size, nb_teams=1, nb_connect=nb_players, seed=seed)
        logger.info("EnvironmentRandomGroup initialized | nb_players=%s size=%s", nb_players, size)

    def _setup(self):
        for i in range(self.nb_players):
            level = random.randint(1, 7)
            self.add_player(0, level)
        logger.debug("EnvironmentRandomGroup setup | players=%s", len(self.players))

    def shuffle_params(self):
        """Shuffle parameters for EnvironmentRandomGroup"""
        self.nb_players = random.randint(2, 10)
        super().shuffle_param(shuffle_size=True)
        logger.debug("EnvironmentRandomGroup params shuffled | nb_players=%s size=%s", self.nb_players, self.size)

    def _compute_reward(self, player, prev_state):
        reward = super()._compute_reward(player, prev_state)
        if player.level > 1 and player.cur_cmd in [player.set, player.broadcast]:
            reward += RewardConstants.HELPING_REWARD
        return reward
