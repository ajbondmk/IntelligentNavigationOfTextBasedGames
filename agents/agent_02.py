import numpy as np
import textworld
import random
import string
import torch
import torch.nn as nn
import torch.optim as optim
from .nn_module import Model


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
        
        self.model = Model(len(all_words), len(all_words), 128, len(self.commands))
        
        self.loss_function = nn.NLLLoss()
        self.optimiser = optim.SGD(self.model.parameters(), lr=0.1)


    def reset(self, env):
        env.activate_state_tracking()
        env.compute_intermediate_reward()


    def act(self, game_state):
        self.model.zero_grad()
        self.model.init_hidden()
        input = self.prepare_sequence(game_state.description)
        output = self.model(input)

        _,b = torch.max(output[-1],0)
        command = self.commands[b]

        print(output[-1])
        print(command)
        print()

        desired_output = torch.ones(8, dtype=torch.long)
        print(output)
        print(desired_output)
        self.finish_step(output, desired_output)

        return command


    # def finish_step(self, game_state):
    def finish_step(self, output, desired_output):
        loss = self.loss_function(output, desired_output)
        loss.backward()
        self.optimiser.step()


    # def finish_episode(self, game_state, done):
    #     reward = 0
    #     if done:
    #         reward = 1
        # self.representation_generator.?


    def prepare_sequence(self, input_string):
        
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
