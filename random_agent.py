""" File containing RandomAgent. """


import random
import textworld
import string


class RandomAgent(textworld.Agent):
    """ Agent that randomly selects actions. """


    # Initialise variables to track test results.
    # self.num_moves_results = []
    score_results = []


    def act(self, game_state):
        """ Choose a random action. """

        description = game_state.description

        actions = []

        if 'north' in description:
            actions.append('go north')
        if 'east' in description:
            actions.append('go east')
        if 'south' in description:
            actions.append('go south')
        if 'west' in description:
            actions.append('go west')
        if 'coin' in description:
            actions.append('take coin')

        return random.choice(actions)
