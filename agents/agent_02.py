import numpy as np
import textworld
import random
import string
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .nn_module import Model
from collections import namedtuple


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


# TODO: Add description.
class Agent02(textworld.Agent):


    def __init__(self, total_count):

        self.actions = ['go north', 'go east', 'go south', 'go west', 'take coin']
        self.action_to_index = {}
        for i in range(len(self.actions)):
            self.action_to_index[self.actions[i]] = i

        # Read all possible words into an array.
        f = open('word_lists/all_words.txt', 'r')
        all_words = f.read().split('\n')
        f.close()
        
        # Create dictionary of each word to a unique index.
        self.word_to_index = {}
        for i in range(len(all_words)):
            self.word_to_index[all_words[i]] = i
        
        # TODO: Investigate hyperparameters.
        self.model = Model(len(all_words), 128, 128, len(self.actions))

        self.memory = ReplayMemory(100, 20)
        
        self.loss_criterion = nn.MSELoss()
        self.optimiser = optim.RMSprop(self.model.parameters())

        self.total_count = total_count - 1
        self.current_count = -1


    def reset(self, env):
        env.activate_state_tracking()
        env.compute_intermediate_reward()
        self.current_count = self.current_count + 1


    def act(self, game_state):

        # print()
        epsilon = (self.total_count - self.current_count) / self.total_count
        # print("Epsilon: ", epsilon)
        if random.random() > epsilon:
            self.model.zero_grad()
            self.model.init_hidden()
            input = self.prepare_input(game_state.description)
            output = self.model(input)
            _,b = torch.max(output,0)
            action = self.actions[b]
            # print("Output:  ", output)
        else:
            action = random.choice(self.actions)
        # print("Action:  ", action)
        return action


    def optimise(self):
        
        batch = self.memory.get_batch()
        if batch is None:
            return

        action_batch = [self.action_to_index[a] for a in batch.action]
        reward_batch = torch.stack([torch.tensor(r, dtype=torch.float) for r in batch.reward])

        action_values = torch.zeros(self.memory.batch_size)
        for i in range(self.memory.batch_size):
            state = batch.state[i]
            self.model.zero_grad()
            self.model.init_hidden()
            action_values[i] = self.model(self.prepare_input(state))[action_batch[i]]

        next_state_values = torch.zeros(self.memory.batch_size)
        for i in range(self.memory.batch_size):
            next_state = batch.next_state[i]
            if next_state is not "":
                self.model.zero_grad()
                self.model.init_hidden()
                next_state_values[i] = torch.max(self.model(self.prepare_input(next_state)))
        
        # TODO: Choose gamma.
        gamma = 0.1
        expected_action_values = reward_batch + (next_state_values * gamma)

        loss = self.loss_criterion(action_values, expected_action_values.detach())
        # print("Loss:    ", loss.item())

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()


    def prepare_input(self, input_string):
        
        # Split the input description text into lowercase words with no punctuation.
        translator = str.maketrans('', '', string.punctuation)
        sanitised_description = input_string.translate(translator).lower()
        words = sanitised_description.split()

        # Remove all stop words and unknown words.
        # Collect Tensors of word indices.
        indices = []
        for word in words:
            if word in self.word_to_index:
                indices.append(self.word_to_index[word])
        
        return torch.tensor(indices, dtype=torch.long)



# TODO: Put in another file?
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

    # def __len__(self):
    #     return len(self.store)

    # def print(self):
    #     print(self.store)
