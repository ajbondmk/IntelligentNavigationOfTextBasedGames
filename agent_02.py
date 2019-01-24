import numpy as np
import textworld
import random
import string
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from nn_module import Model
from replay_memory import PriorityReplayMemory
import sys


class Agent02(textworld.Agent):
    """ Agent that uses an epsilon-greedy policy and a neural network model to select actions. """


    def __init__(self):
        """ Initialise the agent. """

        # Create a list of possible actions.
        self.actions = ['go north', 'go east', 'go south', 'go west', 'take coin']

        # Create a dictionary from each action to a unique index.
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

        # Set the number of words to be input into the model for each input string.
        # If the actual number of words is smaller, all-zero padding will be added.
        # If the actual number of words is larger, the extra words will be ignored.
        # Note: Longest observed sequence had length 41.
        self.num_input_words = 50
        
        # Create a neural network model.
        self.model = Model(len(all_words), self.num_input_words, len(self.actions))

        # Create a memory for transitions.
        self.memory = PriorityReplayMemory(500000, 32, 0.25)
        
        # Create a criterion for calculating loss and an optimiser for training the model.
        self.loss_criterion = nn.MSELoss()
        self.optimiser = optim.Adam(self.model.parameters(), lr=0.001)

        # Initialise epsilon.
        self.epsilon = 0.1


    def set_epsilon(self, epsilon):
        """ Set the value of epsilon, which controls how likely the agent is to choose a random action. """
        self.epsilon = epsilon


    def act(self, game_state):
        """ Choose an action. """
        
        # With probability (1 - epsilon), choose an action using the model.
        if random.random() > self.epsilon:
            self.model.init_hidden(1)
            input = self.encode_inputs([game_state.command], [game_state.description])
            output = self.model(input)[0]
            _,b = torch.max(output,0)
            action = self.actions[b]
            # TODO: Remove this: print("[{:s}]".format(", ".join(str(i) for i in output.tolist())))
        
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
        
        # Create a list of actions for the batch.
        action_value_batch = [self.action_to_index[a] for a in batch.action]

        # Create a tensor of rewards for the batch.
        reward_batch = torch.stack([torch.tensor(r, dtype=torch.float) for r in batch.reward])

        # Calculate the value predicted by the model for each transition in the batch.
        self.model.init_hidden(self.memory.batch_size)
        all_action_values = self.model(self.encode_inputs(batch.action, batch.state))
        action_values = torch.stack([all_action_values[i,action_value_batch[i]] for i in range(len(all_action_values))])

        # Calculate the maximum value predicted by the model for an action taken in the next state of each transition in the batch.
        non_final_next_state_mask = list(map(lambda s: s is not "", batch.next_state))
        non_final_next_states = np.array(batch.next_state)[non_final_next_state_mask]
        non_final_actions = np.array(batch.action)[non_final_next_state_mask]
        non_final_nexts = self.encode_inputs(non_final_actions, non_final_next_states)
        self.model.init_hidden(len(non_final_nexts))
        non_final_next_state_values = self.model(non_final_nexts)
        next_state_values = torch.zeros(self.memory.batch_size)
        next_state_values[torch.tensor(tuple(non_final_next_state_mask), dtype=torch.uint8)] = torch.stack([torch.max(values) for values in non_final_next_state_values])
        
        # Calculate the expected action values for each transition.
        gamma = 0.5
        expected_action_values = reward_batch + (next_state_values * gamma)

        # Calculate loss and optimise the model accordingly.
        loss = self.loss_criterion(action_values, expected_action_values.detach())
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()


    def encode_inputs(self, actions, descriptions):
        """ Takes in a batch of actions and room descriptions. Processes these inputs into a batch of tensors containing the one-hot encodings for each (relevant) word in the pairs of action and description, ready to be input into the model. """
        
        batch_size = min(len(actions), len(descriptions))

        # Create an all-zeros vector of size (batch size, number of words, vocab size)
        encoded_inputs = torch.zeros(batch_size, self.num_input_words, len(self.word_to_index))

        # Repeat for each pair in the batch.
        for i in range(batch_size):

            # Concatenate the action and the description.
            actions = list(map(lambda a : a if a is not None else "", actions))
            concatenated = actions[i] + descriptions[i]

            # Split the concatenated string into lowercase words with no punctuation.
            translator = str.maketrans('', '', string.punctuation)
            sanitised_description = concatenated.translate(translator).lower()
            words = sanitised_description.split()

            # For all words, ignoring all stop words and unknown words, set the relevant 'ones' in the encoding tensor to create one-hot encodings.
            word_num = 0
            for word in words:
                if word in self.word_to_index:
                    encoded_inputs[i, word_num, self.word_to_index[word]] = 1
                    word_num = word_num + 1 

        return encoded_inputs
