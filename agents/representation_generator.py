import string
import torch
import torch.nn as nn


# TODO: Add description.
class RepresentationGenerator():


    def __init__(self):

        # Read all possible words into an array.
        f = open('all_words.txt', 'r')
        words = f.read().split('\n')
        f.close()
        
        # Create LSTM module.
        self.input_dim = len(words)
        self.hidden_dim = 5   # Number of commands.
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim)
        
        # Create dictionary of all words to one-hot encoded vectors.
        self.word_to_vec = {}
        for i in range(self.input_dim):
            vec = torch.zeros(1,1,self.input_dim)
            vec[0,0,i] = 1
            self.word_to_vec[words[i]] = vec


    def split_input(self, input_string):
        
        # Initialise the hidden state.
        # TEMP: The axes semantics are (num_layers, minibatch_size, hidden_dim).
        # TEMP: Alternatively could use `torch.randn`.
        hidden = (torch.zeros(1, 1, self.hidden_dim), torch.zeros(1, 1, self.hidden_dim))

        # Split the input description text into lowercase words with no punctuation.
        translator = str.maketrans('', '', string.punctuation)
        sanitised_description = input_string.translate(translator).lower()
        input_words = sanitised_description.split()

        # For each word, pass the vector into the LSTM module.
        for input_word in input_words:
            if input_word in self.word_to_vec:
                out, hidden = self.lstm(self.word_to_vec[input_word], hidden)
                # print(out)
                # print(hidden)
                # print()
