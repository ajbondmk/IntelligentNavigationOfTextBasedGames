import numpy as np
import textworld
import random


class RandomAgent(textworld.Agent):
    """ Agent that randomly selects commands. """


    # A list of all possible commands.
    commands = ['go north', 'go east', 'go south', 'go west', 'take coin']


    def act(self, game_state):
        """ Choose a random action. """
        return random.choice(self.commands)
