from server.environment.environment import Environment
from server.environment.reward_constant import RewardConstants
from server import constant
import random
from server.logging_config import get_logger

logger = get_logger(__name__)


class EnvironmentStrictIncantation(Environment):
    """
    Strict rules: agents must coordinate perfectly for incantation. Ends when incantation is done or resources run out.
    Difficulty: ⭐⭐⭐⭐⭐
    """
    def __init__(self, level=2, size=8, seed=None, **kwargs):
        self.level = level
        self.nb_players = max(2, constant.ElevationRequirement.requirements[level][0])
        self.incantation_started = False
        self.incantation_tile = None
        self.incantation_participants = set()
        self.initial_positions = {}
        super().__init__(size=size, nb_teams=1, nb_connect=self.nb_players, seed=seed)
        logger.info("EnvironmentStrictIncantation initialized | level=%s nb_players=%s size=%s", level, self.nb_players, size)

    def _setup(self):
        center = self.size // 2
        tile = self.map.tiles[center][center]
        self.incantation_tile = (center, center)
        _, resources = constant.ElevationRequirement.requirements[self.level]
        for resource, count in resources.items():
            tile[resource] = count
        for i in range(self.nb_players):
            player = self.add_player(0, self.level, (center, center))
            self.initial_positions[player.id] = (center, center)
        logger.debug("EnvironmentStrictIncantation setup | players=%s center=%s", len(self.players), (center, center))

    def _compute_reward(self, player, prev_state):
        reward = super()._compute_reward(player, prev_state)
        incantation_active = self.incantation_tile in self.map.incanted_tile
        if self.incantation_started or incantation_active:
            if player.position != prev_state.get("position", player.position):
                reward += RewardConstants.MOVEMENT_DURING_INCANTATION_PENALTY
            if player.cur_cmd in [player.forward, player.left, player.right]:
                reward += RewardConstants.MOVEMENT_COMMAND_PENALTY
            if player.position != self.incantation_tile:
                reward += RewardConstants.ABANDONING_INCANTATION_PENALTY
        if not self.incantation_started and player.cur_cmd == player.incantation:
            players_on_tile = [p for p in self.players \
                             if p.position == self.incantation_tile and p.level == self.level]
            if len(players_on_tile) >= self.nb_players:
                reward += RewardConstants.INCANTATION_START_REWARD * 2
                self.incantation_started = True
                self.incantation_participants = {p.id for p in players_on_tile}
        if player.level > prev_state["level"]:
            reward += RewardConstants.LEVEL_PROGRESSION_BASE * 3
        if incantation_active and player.position == self.incantation_tile:
            reward += RewardConstants.INCANTATION_DISCIPLINE_REWARD
        return reward

    def _check_termination(self):
        if any(p.level > self.level for p in self.players):
            logger.info("EnvironmentStrictIncantation terminated: player leveled up")
            return True
        if self.incantation_started and self.incantation_tile not in self.map.incanted_tile:
            if not any(p.level > self.level for p in self.players):
                logger.info("EnvironmentStrictIncantation terminated: incantation resolved without level up")
                return True
        if self.incantation_started:
            players_on_tile = [p for p in self.players \
                             if p.position == self.incantation_tile and p.level >= self.level]
            if len(players_on_tile) < self.nb_players:
                logger.info("EnvironmentStrictIncantation terminated: participants left the tile")
                return True
        if self.server.ticks > 1000:
            logger.info("EnvironmentStrictIncantation terminated: tick limit reached")
            return True
        return False

    def shuffle_params(self):
        """Shuffle parameters for EnvironmentStrictIncantation"""
        self.level = random.randint(2, 7)
        super().__init__(size=self.size, nb_teams=1, nb_connect=self.nb_players, seed=random.randint(1, 1000))
        logger.debug("EnvironmentStrictIncantation params shuffled | level=%s size=%s nb_players=%s", self.level, self.size, self.nb_players)
