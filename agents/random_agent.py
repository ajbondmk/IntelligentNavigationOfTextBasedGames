import numpy as np
import textworld
import random

""" Agent that randomly selects commands. """
class RandomAgent(textworld.Agent):

    commands = ['go north', 'go east', 'go south', 'go west', 'take coin']

    def reset(self, env):
        env.activate_state_tracking()
        env.compute_intermediate_reward()

    def act(self, game_state):
        return random.choice(self.commands)
