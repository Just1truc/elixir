import unittest
from server.environment.alone_eggs.env import EnvironmentAloneWithConnectSlot

class TestEnvironmentAloneEggs(unittest.TestCase):
    def test_add_player(self):
        env = EnvironmentAloneWithConnectSlot(size=10)
        env.add_player(0)
        self.assertIsNotNone(env.players[1])
        self.assertEqual(len(env.players), 2)

    def test_clone(self):
        env = EnvironmentAloneWithConnectSlot(size=10)
        clone = env.clone()
        self.assertNotEqual(env, clone)
