import numpy as np
import textworld
import random

class RandomAgent(textworld.Agent):

    commands = ['go north', 'go east', 'go south', 'go west', 'take coin']

    """ Agent that randomly selects commands from the admissible ones. """
    def __init__(self):
        self.rng = random.Random()

    def reset(self, env):
        env.activate_state_tracking()
        env.compute_intermediate_reward()

    def act(self, game_state, reward, done):
        return self.rng.choice(self.commands)
    