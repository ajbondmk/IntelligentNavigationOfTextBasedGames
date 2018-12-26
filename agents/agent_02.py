import numpy as np
import textworld
import random
import string
import torch
import torch.nn as nn
import torch.optim as optim
from .nn_module import Model
from collections import namedtuple


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


# TODO: Add description.
class Agent02(textworld.Agent):


    def __init__(self):
        self.rng = random.Random()

        self.commands = ['go north', 'go east', 'go south', 'go west', 'take coin']

        # Read all possible words into an array.
        f = open('word_lists/all_words.txt', 'r')
        all_words = f.read().split('\n')
        f.close()
        
        # Create dictionary of each word to a unique index.
        self.word_to_index = {}
        for i in range(len(all_words)):
            self.word_to_index[all_words[i]] = i
        
        # TODO: Investigate hyperparameters.
        self.model = Model(len(all_words), 128, 128, len(self.commands))

        self.memory = ReplayMemory(5, 3)
        
        # self.loss_function = nn.NLLLoss()
        # self.optimiser = optim.Adam(self.model.parameters(), lr=0.1)


    def reset(self, env):
        env.activate_state_tracking()
        env.compute_intermediate_reward()


    def act(self, game_state):
        self.model.zero_grad()
        self.model.init_hidden()
        input = self.prepare_input(game_state.description)
        output = self.model(input)

        _,b = torch.max(output[-1],0)
        command = self.commands[b]

        print(output[-1])
        # self.rng = random.Random()
        # command = self.rng.choice(self.commands)
        print(command)

        return command


    def optimise(self):
        
        batch = self.memory.get_batch()
        if batch is None:
            return
        
        # START HERE
        

        # self.optimiser.zero_grad()
        # self.loss.backward()
        # for parameter in policy_net.parameters():
        #     parameter.grad.data.clamp_(-1, 1)
        # self.optimiser.step()


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
