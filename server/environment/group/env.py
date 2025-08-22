from server.environment.environment import Environment
from server.environment.reward_constant import RewardConstants

import random
from server.logging_config import get_logger

logger = get_logger(__name__)


class EnvironmentGroup(Environment):
    """
    Multiple agents cooperate to collect resources and survive. Ends when all players die.
    Difficulty: ⭐⭐⭐
    """
    def __init__(self, level, nb_players, size=10, seed=None, **kwargs):
        self.level = level
        self.nb_players = nb_players
        super().__init__(size=size, nb_teams=1, nb_connect=nb_players, seed=seed)
        logger.info("EnvironmentGroup initialized | level=%s nb_players=%s size=%s", level, nb_players, size)

    def _setup(self):
        center = self.size // 2
        for _ in range(self.nb_players):
            x = center + random.randint(-1, 1)
            y = center + random.randint(-1, 1)
            self.add_player(0, self.level, (x, y))
        logger.debug("EnvironmentGroup setup | players=%s center=%s", len(self.players), (center, center))

    def _compute_reward(self, player, prev_state):
        reward = super()._compute_reward(player, prev_state)
        if player.last_command == player.broadcast:
            reward += RewardConstants.BROADCAST_REWARD
        teammate_near = any(
            p != player and abs(p.position[0]-player.position[0]) <= 1 \
            and abs(p.position[1]-player.position[1]) <= 1 \
            for p in self.players
        )
        if teammate_near:
            reward += RewardConstants.TEAMMATE_PROXIMITY_REWARD
        return reward

    def _check_termination(self):
        for player in self.players:
            if player.level >= self.level + 1:
                logger.info("EnvironmentGroup terminated: player leveled up")
                return True
        return False

    def shuffle_params(self):
        """Shuffle parameters for EnvironmentGroup"""
        self.level = random.randint(1, 6)
        self.nb_players = random.randint(2, 8)
        super().shuffle_param(shuffle_size=True)
        logger.debug("EnvironmentGroup params shuffled | level=%s nb_players=%s size=%s", self.level, self.nb_players, self.size)
