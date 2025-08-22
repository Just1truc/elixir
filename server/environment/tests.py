import unittest

from server.environment.alone.env import EnvironmentAlone
from server.environment.alone_eggs.env import EnvironmentAloneWithConnectSlot
from server.environment.group.env import EnvironmentGroup
from server.environment.random_group.env import EnvironmentRandomGroup
from server.environment.survival.env import EnvironmentSurvival
from server.environment.incantation.env import EnvironmentIncantation
from server.environment.close_incantation.env import EnvironmentCloseToIncantation
from server.environment.strict_incantation.env import EnvironmentStrictIncantation


class TestEnvironmentCommon(unittest.TestCase):
    def test_clone_creates_deep_copy(self):
        env = EnvironmentAloneWithConnectSlot(size=10, seed=99)
        env.reset()
        # Ensure at least one player
        if len(env.players) == 0:
            env.add_player(0)
        clone = env.clone()
        # Different objects
        self.assertIsNot(env, clone)
        self.assertIsNot(env.server, clone.server)
        self.assertEqual(env.size, clone.size)
        self.assertEqual(len(env.players), len(clone.players))
        # Mutate original player's inventory; clone should not change
        p0 = env.players[0]
        p0.inventory = {**p0.inventory, 'food': (p0.inventory.get('food', 0) + 5)}
        c0 = clone.players[0]
        self.assertNotEqual(p0.inventory.get('food', 0), c0.inventory.get('food', 0))


if __name__ == '__main__':
    unittest.main()
