import numpy as np
import textworld
import random
import string
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from nn_module import Model
from replay_memory import ReplayMemory
from debug_print import debug_print


class Agent02(textworld.Agent):
    """ Agent that uses an epsilon-greedy policy and a neural network model to select actions. """


    def __init__(self):
        """ Initialise the agent. """

        # Create a list of possible actions.
        self.actions = ['go north', 'go east', 'go south', 'go west', 'take coin']

        # Create a dictionary from each command to a unique index.
        self.action_to_index = {}
        for i in range(len(self.actions)):
            self.action_to_index[self.actions[i]] = i

        # Read all possible words into an array.
        f = open('word_lists/all_words.txt', 'r')
        all_words = f.read().split('\n')
        f.close()
        
        # Create dictionary from each word to a unique index.
        self.word_to_index = {}
        for i in range(len(all_words)):
            self.word_to_index[all_words[i]] = i
        
        # TODO: Choose hyperparameters.
        # Create a neural network model.
        self.model = Model(len(all_words), 128, 128, len(self.actions))

        # TODO: Choose memory and batch size.
        # Create a memory for transitions.
        self.memory = ReplayMemory(100, 20)
        
        # Create a criterion for calculating loss and an optimiser for training the model.
        self.loss_criterion = nn.MSELoss()
        self.optimiser = optim.RMSprop(self.model.parameters())

        # Initialise epsilon.
        self.epsilon = 0.1


    def set_epsilon(self, epsilon):
        """ Set the value of epsilon, which controls how likely the agent is to choose a random action. """
        self.epsilon = epsilon
        debug_print("Epsilon:  {:f}".format(self.epsilon))


    def act(self, game_state):
        """ Choose an action. """

        debug_print()
        
        # With probability (1 - epsilon), choose an action using the model.
        if random.random() > self.epsilon:
            self.model.zero_grad()
            self.model.init_hidden()
            input = self.prepare_input(game_state.description)
            output = self.model(input)
            _,b = torch.max(output,0)
            action = self.actions[b]
            debug_print("Output:   [{:s}]".format(", ".join(str(i) for i in output.tolist())))
        
        # With probability epsilon, choose a random action.
        else:
            action = random.choice(self.actions)
        
        return action


    def optimise(self):
        """ Train the model based on a batch of transitions from the memory. """
        
        # If there are not enough transitions in memory to form a batch, no training is performed.
        batch = self.memory.get_batch()
        if batch is None:
            return

        # Create a list of actions and a tensor of rewards.
        action_batch = [self.action_to_index[a] for a in batch.action]
        reward_batch = torch.stack([torch.tensor(r, dtype=torch.float) for r in batch.reward])

        # Calculate the value predicted by the model for the action taken in each transition.
        action_values = torch.zeros(self.memory.batch_size)
        for i in range(self.memory.batch_size):
            state = batch.state[i]
            self.model.zero_grad()
            self.model.init_hidden()
            action_values[i] = self.model(self.prepare_input(state))[action_batch[i]]

        # Calculate the maximum value predicted by the model for an action taken in the next state of each transition.
        next_state_values = torch.zeros(self.memory.batch_size)
        for i in range(self.memory.batch_size):
            next_state = batch.next_state[i]
            if next_state is not "":
                self.model.zero_grad()
                self.model.init_hidden()
                next_state_values[i] = torch.max(self.model(self.prepare_input(next_state)))
        
        # Calculate the expected action values for each transition.
        # TODO: Choose gamma.
        gamma = 0.1
        expected_action_values = reward_batch + (next_state_values * gamma)

        # Calculate loss and optimise the model accordingly.
        loss = self.loss_criterion(action_values, expected_action_values.detach())
        debug_print("Loss:     {:f}".format(loss.item()))
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()


    def prepare_input(self, input_string):
        """ Process a room description string into a format to be input into the model. """
        
        # Split the input description text into lowercase words with no punctuation.
        translator = str.maketrans('', '', string.punctuation)
        sanitised_description = input_string.translate(translator).lower()
        words = sanitised_description.split()

        # Collect Tensors of word indices, removing all stop words and unknown words.
        indices = []
        for word in words:
            if word in self.word_to_index:
                indices.append(self.word_to_index[word])
        
        return torch.tensor(indices, dtype=torch.long)
