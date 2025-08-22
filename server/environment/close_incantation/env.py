from server.environment.incantation.env import EnvironmentIncantation
from server.environment.reward_constant import RewardConstants
from server import constant
import random
from server.logging_config import get_logger

logger = get_logger(__name__)


class EnvironmentCloseToIncantation(EnvironmentIncantation):
    """
    Agents start near incantation tile, must coordinate and manage resources. Ends when incantation is done or resources run out.
    Difficulty: ⭐⭐⭐⭐
    """
    def __init__(self, level, size=10, seed=None, **kwargs):
        super().__init__(size=size, level=level, seed=seed)
        logger.info("EnvironmentCloseToIncantation initialized | level=%s size=%s", level, size)

    def _setup(self):
        center = self.size // 2
        tile = self.map.tiles[center][center]
        _, resources = constant.ElevationRequirement.requirements[self.level]
        for resource, count in resources.items():
            tile[resource.value] = max(0, count - 1)
        for resource, count in resources.items():
            if count > 0:
                x = (center + 1) % self.size
                y = center
                self.map.tiles[y][x][resource.value] = 1
        for i in range(self.nb_players):
            self.add_player(0, self.level, (center, center))
        logger.debug("EnvironmentCloseToIncantation setup | players=%s center=%s", len(self.players), (center, center))

    def _compute_reward(self, player, prev_state):
        reward = super()._compute_reward(player, prev_state)
        _, resources_req = constant.ElevationRequirement.requirements[self.level]
        for resource, count in resources_req.items():
            prev_count = prev_state["inventory"].get(resource.value, 0)
            current_count = player.inventory.get(resource.value, 0)
            if current_count > prev_count:
                reward += RewardConstants.RESOURCE_COLLECTION_REWARD
        center = self.size // 2
        if player.position == (center, center) and player.cur_cmd == player.set:
            reward += RewardConstants.RESOURCE_DELIVERY_REWARD
        return reward

    def shuffle_custom_params(self):
        """Shuffle parameters specific to EnvironmentCloseToIncantation"""
        self.level = random.randint(2, 7)  # Levels 2-7 have interesting requirements
        self.size = random.randrange(8, 25)
        super().shuffle_params(shuffle_size=True)
        logger.debug("EnvironmentCloseToIncantation params shuffled | level=%s size=%s", self.level, self.size)
