import unittest
from server.environment.strict_incantation.env import EnvironmentStrictIncantation
from server import constant

class TestStrictIncantationEnvironment(unittest.TestCase):
    def setUp(self):
        self.env = EnvironmentStrictIncantation(level=2, size=8, seed=42)

    def test_environment_setup(self):
        state = self.env.reset()
        self.assertEqual(self.env.level, 2)
        self.assertEqual(self.env.size, 8)
        self.assertGreaterEqual(self.env.nb_players, 2)
        center = self.env.size // 2
        self.assertEqual(self.env.incantation_tile, (center, center))
        _, required_resources = constant.ElevationRequirement.requirements[2]
        tile = self.env.map.tiles[center][center]
        for resource, count in required_resources.items():
            self.assertGreaterEqual(tile.get(resource, 0), count)
        players = state["players"]
        for player in players:
            self.assertEqual(player.position, (center, center))
            self.assertEqual(player.level, 2)

    def test_incantation_start_tracking(self):
        state = self.env.reset()
        players = state["players"]
        self.assertFalse(self.env.incantation_started)
        players[0].add_cmd("Incantation")
        for _ in range(10):
            state, rewards, done = self.env.step()
            if self.env.incantation_started:
                break

    def test_movement_penalty_system(self):
        state = self.env.reset()
        players = state["players"]
        self.env.incantation_started = True
        players[0].add_cmd("Forward")
        state, rewards, done = self.env.step()
        if rewards:
            has_penalty = any(reward < 0 for reward in rewards)

    def test_termination_conditions(self):
        state = self.env.reset()
        self.env.server.ticks = 1001
        state, rewards, done = self.env.step()
        self.assertTrue(done)
        self.env.reset()
        players = self.env.players
        players[0].level = self.env.level + 1
        terminated = self.env._check_termination()
        self.assertTrue(terminated)
        
    def test_reward_constants_exist(self):
        from server.environment.reward_constant import RewardConstants
        self.assertTrue(hasattr(RewardConstants, 'MOVEMENT_DURING_INCANTATION_PENALTY'))
        self.assertTrue(hasattr(RewardConstants, 'MOVEMENT_COMMAND_PENALTY'))
        self.assertTrue(hasattr(RewardConstants, 'ABANDONING_INCANTATION_PENALTY'))
        self.assertTrue(hasattr(RewardConstants, 'INCANTATION_DISCIPLINE_REWARD'))
