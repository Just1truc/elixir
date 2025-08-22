import unittest
from server.environment.survival.env import EnvironmentSurvival

class TestEnvironmentSurvival(unittest.TestCase):
    def test_survival_environment(self):
        env = EnvironmentSurvival(nb_players=2, food_amount=3, size=6)
        state = env.reset()
        self.assertIsNotNone(env)
        self.assertIsNotNone(env.map)
        self.assertEqual(env.size, 6)
        self.assertEqual(len(env.teams_name), 1)
        self.assertEqual(len(state["players"]), 2)
