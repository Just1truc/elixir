"""
my_env.py

Template for creating custom environment classes or logic for the Zappy simulation.
"""
from server.environment.environment import Environment
from server.environment.reward_constant import RewardConstants

class MyEnv(Environment):
    """My custom environment"""
    def __init__(self, size=5, nb_teams=2, nb_connect=5, seed=None, **kwargs):
        super().__init__(size=size, nb_teams=nb_teams, nb_connect=nb_connect, seed=seed, **kwargs)

    def _setup(self):
        """Computed function at initialisation"""
        pass
    
    def _compute_reward(self, player, prev_state):
        """Reward for single player environment"""
        reward = super()._compute_reward(player, prev_state)
        return reward
    
    def _check_termination(self):
        """Terminate when returned true"""
        return False

