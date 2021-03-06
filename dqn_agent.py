""" File containing DQNAgent. """


import random
import string
import numpy as np
import textworld
import torch
import torch.nn as nn
import torch.optim as optim
from nn_module import Model
from replay_memory import PriorityReplayMemory


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DQNAgent(textworld.Agent):
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
        words_file = open('word_lists/all_words.txt', 'r')
        all_words = words_file.read().split('\n')
        words_file.close()

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
        self.model = Model(len(all_words), len(self.actions))
        self.model.to(DEVICE)

        # Create a memory for transitions.
        self.memory = PriorityReplayMemory(500000, 32, 0.25)

        # Create a criterion for calculating loss and an optimiser for training the model.
        self.loss_criterion = nn.MSELoss()
        self.optimiser = optim.Adam(self.model.parameters(), lr=0.001)

        # Initialise epsilon.
        self.epsilon = 0.1

        # Initialise variables to track test results.
        self.num_moves_results = []
        self.score_results = []
        self.num_moves_results_test = []
        self.score_results_test = []


    def set_epsilon(self, epsilon):
        """ Set the value of epsilon, which controls how likely the agent is
        to choose a random action. """
        self.epsilon = epsilon


    def act(self, game_state):
        """ Choose an action. """

        # With probability (1 - epsilon), choose an action using the model.
        if random.random() > self.epsilon:
            self.model.init_hidden(1)
            model_input = self.encode_inputs([game_state.description], [game_state.command])
            model_output = self.model(*model_input)[0]
            _, action_index = torch.max(model_output, 0)
            action = self.actions[action_index]

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
        reward_batch = torch.stack(
            [torch.tensor(r, dtype=torch.float) for r in batch.reward]
        ).to(DEVICE)

        # Calculate the value predicted by the model for each transition in the batch.
        self.model.init_hidden(self.memory.batch_size)
        all_action_values = self.model(*self.encode_inputs(batch.state, batch.action))
        action_values = torch.stack(
            [all_action_values[i, action_value_batch[i]] for i in range(len(all_action_values))])

        # Calculate the maximum value predicted by the model for an action taken in
        # the next state of each transition in the batch.
        non_final_next_state_mask = list(map(lambda s: s != "", batch.next_state))
        non_final_next_states = np.array(batch.next_state)[non_final_next_state_mask]
        non_final_actions = np.array(batch.action)[non_final_next_state_mask]
        non_final_nexts, input_lengths = self.encode_inputs(
            non_final_next_states, non_final_actions)
        self.model.init_hidden(len(non_final_nexts))
        non_final_next_state_values = self.model(
            non_final_nexts.to(DEVICE), input_lengths.to(DEVICE))
        next_state_values = torch.zeros(self.memory.batch_size).to(DEVICE)
        next_state_values[
            torch.tensor(tuple(non_final_next_state_mask), dtype=torch.uint8).to(DEVICE)
        ] = torch.stack([torch.max(values) for values in non_final_next_state_values])

        # Calculate the expected action values for each transition.
        gamma = 0.5
        expected_action_values = reward_batch + (next_state_values * gamma)

        # Calculate loss and optimise the model accordingly.
        loss = self.loss_criterion(action_values, expected_action_values.detach())
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()


    def encode_inputs(self, descriptions, actions):
        """ Takes in a batch of actions and room descriptions. Processes these inputs into
        a batch of tensors containing the one-hot encodings for each (relevant) word in the
        pairs of action and description, ready to be input into the model. """

        batch_size = min(len(descriptions), len(actions))

        # Create an all-zeros vector of size (batch size, number of words, vocab size).
        encoded_inputs = torch.zeros(batch_size, self.num_input_words, len(self.word_to_index))

        # Initialise the vector of input lengths.
        input_lengths = torch.zeros(batch_size, dtype=torch.long)

        # Repeat for each pair in the batch.
        for i in range(batch_size):

            # Concatenate the action and the description.
            actions = list(map(lambda a: a if a is not None else "", actions))
            concatenated = descriptions[i] + actions[i]

            # Split the concatenated string into lowercase words with no punctuation.
            translator = str.maketrans('', '', string.punctuation)
            sanitised_description = concatenated.translate(translator).lower()
            words = sanitised_description.split()

            # For all words, ignoring all stop words and unknown words, set the
            # relevant 'ones' in the encoding tensor to create one-hot encodings.
            word_num = 0
            for word in words:
                if word in self.word_to_index:
                    encoded_inputs[i, word_num, self.word_to_index[word]] = 1
                    word_num = word_num + 1
                    if word_num == self.num_input_words:
                        break

            # Add to the lengths array.
            input_lengths[i] = word_num

        return (encoded_inputs.to(DEVICE), input_lengths.to(DEVICE))
