import random
from collections import namedtuple


# An object which represents a transition from one state to the next_state, via an action, and receiving a reward.
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    """ A fixed-length memory which stores the latest transitions. """


    def __init__(self, max_length, batch_size):
        """ Initialise the memory. """
        self.max_length = max_length
        self.batch_size = batch_size
        self.store = []
        self.current_position = 0


    def add_item(self, *item):
        """ Add a new transition memory, overwriting the oldest transition if necessary. """
        if len(self.store) < self.max_length:
            self.store.append(None)
        self.store[self.current_position] = Transition(*item)
        self.current_position = (self.current_position + 1) % self.max_length


    def get_batch(self):
        """
        If memory contains enough transitions, select a random batch of them.
        Instead of returning a batch of transitions, return a transition of batches.
        """
        if len(self.store) < self.batch_size:
            return None
        return Transition(*zip(*random.sample(self.store, self.batch_size)))
