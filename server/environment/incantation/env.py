from server.environment.environment import Environment
from server.environment.reward_constant import RewardConstants
from server import constant
import random
from server.logging_config import get_logger

logger = get_logger(__name__)


class EnvironmentIncantation(Environment):
    """
    Agents must gather resources and coordinate to perform incantation. Ends when incantation is done or resources run out.
    Difficulty: ⭐⭐⭐⭐⭐
    """
    def __init__(self, level, size=10, seed=None, **kwargs):
        self.level = level
        self.nb_players = constant.ElevationRequirement.requirements[level][0]
        super().__init__(size=size, nb_teams=1, seed=seed)
        logger.info("EnvironmentIncantation initialized | level=%s size=%s nb_players=%s", level, size, self.nb_players)

    def _setup(self):
        center = self.size // 2
        tile = self.map.tiles[center][center]
        _, resources = constant.ElevationRequirement.requirements[self.level]
        for resource, count in resources.items():
            tile[resource.value] = count
        for i in range(self.nb_players):
            self.add_player(0, self.level, (center, center))
        logger.debug("EnvironmentIncantation setup | players=%s center=%s", len(self.players), (center, center))

    def _compute_reward(self, player, prev_state):
        reward = super()._compute_reward(player, prev_state)
        if player.cur_cmd == player.incantation:
            reward += RewardConstants.INCANTATION_START_REWARD
        _, resources_req = constant.ElevationRequirement.requirements[self.level]
        has_resources = all(
            self.map.tiles[player.position[1]][player.position[0]].get(res.value, 0) >= count
            for res, count in resources_req.items()
        )
        if has_resources:
            reward += RewardConstants.RESOURCE_READINESS_REWARD
        return reward

    def _check_termination(self):
        center = self.size // 2
        tile = self.map.tiles[center][center]
        if any(p.cur_cmd == p.incantation for p in self.players):
            logger.info("EnvironmentIncantation terminated: incantation started")
            return True
        _, resources_req = constant.ElevationRequirement.requirements[self.level]
        missing_resource = any(
            tile.get(res.value, 0) < count
            for res, count in resources_req.items()
        )
        if missing_resource:
            logger.info("EnvironmentIncantation terminated: missing resources on tile")
            return True
        return False

    def shuffle_params(self):
        """Shuffle parameters for EnvironmentIncantation"""
        self.level = random.randint(2, 7)
        super().shuffle_param(shuffle_size=True)
        logger.debug("EnvironmentIncantation params shuffled | level=%s size=%s", self.level, self.size)
