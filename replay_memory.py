import random
from collections import namedtuple


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, max_length, batch_size):
        self.max_length = max_length
        self.batch_size = batch_size
        self.store = []
        self.current_position = 0

    def add_item(self, *item):
        if len(self.store) < self.max_length:
            self.store.append(None)
        self.store[self.current_position] = Transition(*item)
        self.current_position = (self.current_position + 1) % self.max_length

    def get_batch(self):
        if len(self.store) < self.batch_size:
            return None
        return Transition(*zip(*random.sample(self.store, self.batch_size)))
