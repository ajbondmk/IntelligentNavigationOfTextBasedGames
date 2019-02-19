import numpy as np
import textworld
import random


class RandomAgent(textworld.Agent):
    """ Agent that randomly selects actions. """


    # A list of all possible actions.
    actions = ['go north', 'go east', 'go south', 'go west', 'take coin']

    # Initialise variables to track test results.
    # self.num_moves_results = []
    score_results = []


    def act(self, game_state):
        """ Choose a random action. """
        return random.choice(self.actions)
