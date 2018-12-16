import numpy as np
import textworld
import random
import string
from .representation_generator import RepresentationGenerator

# TODO: Add description.
class Agent01(textworld.Agent):

    commands = ['go north', 'go east', 'go south', 'go west', 'take coin']

    def __init__(self):
        self.rng = random.Random()
        self.representation_generator = RepresentationGenerator()

    def reset(self, env):
        env.activate_state_tracking()
        env.compute_intermediate_reward()

    def act(self, game_state, reward, done):
        self.representation_generator.split_input(game_state.description)
        return self.rng.choice(self.commands)
