import unittest
from server.environment.group.env import EnvironmentGroup

class TestEnvironmentGroup(unittest.TestCase):
    def test_group_environment(self):
        env = EnvironmentGroup(level=2, nb_players=3, size=8)
        state = env.reset()
        self.assertIsNotNone(env)
        self.assertIsNotNone(env.map)
        self.assertEqual(env.size, 8)
        self.assertEqual(len(env.teams_name), 1)
        self.assertEqual(len(state["players"]), 3)
