import numpy as np
import textworld
import random


class RandomAgent(textworld.Agent):
    """ Agent that randomly selects commands. """


    # A list of all possible commands.
    commands = ['go north', 'go east', 'go south', 'go west', 'take coin']


    def reset(self, env):
        """ Reset the agent (should be used before starting a new game). """
        # TODO: This activates state tracking (to find valid commands) and enables intermediate reward computation (to detect if a game is lost). Are either of these necessary?
        env.activate_state_tracking()
        env.compute_intermediate_reward()


    def act(self, game_state):
        """ Choose a random action. """
        return random.choice(self.commands)
