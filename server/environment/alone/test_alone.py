import unittest
from server.environment.alone.env import EnvironmentAlone

class TestEnvironmentAlone(unittest.TestCase):
    def test_alone_environment(self):
        env = EnvironmentAlone(level=1, size=5)
        state = env.reset()
        self.assertIsNotNone(env)
        self.assertIsNotNone(env.map)
        self.assertEqual(env.size, 5)
        self.assertEqual(len(env.teams_name), 1)
        self.assertEqual(len(state["players"]), 1)
