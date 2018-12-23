import numpy as np
import textworld
import random
import string
from .representation_generator import RepresentationGenerator

# TODO: Add description.
class Agent01(textworld.Agent):

    def __init__(self):
        self.rng = random.Random()
        self.representation_generator = RepresentationGenerator()

    def reset(self, env):
        env.activate_state_tracking()
        env.compute_intermediate_reward()

    def act(self, game_state):
        return self.representation_generator.select_action(game_state.description)
