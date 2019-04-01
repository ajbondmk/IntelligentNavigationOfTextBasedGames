""" File containing information to do with the memory of DQNAgent. """


import random
from collections import namedtuple



# An object which represents a transition from one state to the next_state, via an action,
# and receiving a reward.
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))



class PriorityReplayMemory:
    """ A ReplayMemory which is split into priority and non-priority Transitions. """


    def __init__(self, max_length, batch_size, priority_fraction):
        """
        Initialise the memories.
        Priority memory takes up priority_fraction of the batch_size.
        Non-priority memory takes up the remaining space.
        """
        self.batch_size = batch_size
        priority_max_length = round(max_length * priority_fraction)
        non_priority_max_length = max_length - priority_max_length
        self.priority_batch_size = round(batch_size * priority_fraction)
        self.non_priority_batch_size = batch_size - self.priority_batch_size
        self.priority_memory = ReplayMemory(priority_max_length, self.priority_batch_size)
        self.non_priority_memory = ReplayMemory(
            non_priority_max_length, self.non_priority_batch_size)


    def add_item(self, *item):
        """ Add a new transition memory to the relevant memory (priority if reward > 0). """
        if Transition(*item).reward is 0:
            self.non_priority_memory.add_item(*item)
        else:
            self.priority_memory.add_item(*item)


    def get_batch(self):
        """
        If memory contains enough transitions, select a random batch of them from each memory
        (proportionate to priority_fraction) and merge them.
        Instead of returning a batch of transitions, return a transition of batches.
        """
        priority_batch = self.priority_memory.get_batch()
        non_priority_batch = self.non_priority_memory.get_batch()

        if priority_batch is None or non_priority_batch is None:
            return None

        # For each state in a Transition, arrange batch items into a tuple according to
        # random_order where random_order indexes across both priority and non_priority batches.
        # Use these new tuples (one for each state in a Transition) to form a new Transition
        # of batches.
        random_order = list(range(self.batch_size))
        random.shuffle(random_order)
        return Transition(*map(
            lambda i: tuple(map(
                lambda j: non_priority_batch[i][j]
                if j < self.non_priority_batch_size
                else priority_batch[i][j - self.non_priority_batch_size],
                random_order)),
            range(4)))



class ReplayMemory:
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
