import unittest
from server.environment.random_group.env import EnvironmentRandomGroup

class TestEnvironmentRandomGroup(unittest.TestCase):
    def test_random_group_environment(self):
        env = EnvironmentRandomGroup(nb_players=4, size=12)
        state = env.reset()
        self.assertIsNotNone(env)
        self.assertIsNotNone(env.map)
        self.assertEqual(env.size, 12)
        self.assertEqual(len(env.teams_name), 1)
        self.assertEqual(len(state["players"]), 4)
