""" File containing RandomAgent. """


import random
import textworld


class RandomAgent(textworld.Agent):
    """ Agent that randomly selects actions. """


    # A list of all possible actions.
    actions = ['go north', 'go east', 'go south', 'go west', 'take coin']

    # Initialise variables to track test results.
    score_results = []


    def act(self):
        """ Choose a random action. """
        return random.choice(self.actions)
