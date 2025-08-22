import unittest
from server.environment.incantation.env import EnvironmentIncantation
from server import constant

class TestEnvironmentIncantation(unittest.TestCase):
    def test_incantation_environment(self):
        env = EnvironmentIncantation(level=2, size=10)
        state = env.reset()
        self.assertIsNotNone(env)
        self.assertIsNotNone(env.map)
        self.assertEqual(env.size, 10)
        self.assertEqual(len(env.teams_name), 1)
        required_players = constant.ElevationRequirement.requirements[2][0]
        self.assertEqual(len(state["players"]), required_players)
