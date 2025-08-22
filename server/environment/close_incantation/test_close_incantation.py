import unittest
from server.environment.close_incantation.env import EnvironmentCloseToIncantation
from server import constant

class TestEnvironmentCloseIncantation(unittest.TestCase):
    def test_close_incantation_environment(self):
        env = EnvironmentCloseToIncantation(level=3, size=10)
        state = env.reset()
        self.assertIsNotNone(env)
        self.assertIsNotNone(env.map)
        self.assertEqual(env.size, 10)
        self.assertEqual(len(env.teams_name), 1)
        required_players = constant.ElevationRequirement.requirements[3][0]
        self.assertEqual(len(state["players"]), required_players)
