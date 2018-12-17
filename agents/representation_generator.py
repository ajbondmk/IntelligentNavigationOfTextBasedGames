import string
import torch
import torch.nn as nn

# TODO: Add description.
class RepresentationGenerator():

    def __init__(self):
        f = open('all_words.txt', 'r')
        self.all_words = f.read().split('\n')
        f.close()
        
        self.input_dim = len(self.all_words)
        self.hidden_dim = 5   # Number of commands.
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim)

    def split_input(self, input_string):
        
        # The axes semantics are (num_layers, minibatch_size, hidden_dim).
        hidden = (torch.zeros(1, 1, self.hidden_dim), torch.zeros(1, 1, self.hidden_dim))
        # Alternatively could use `torch.randn`.

        translator = str.maketrans('', '', string.punctuation)
        sanitised_description = input_string.translate(translator).lower()
        words = sanitised_description.split()

        for word in words:
            word_vec = torch.zeros(1,1,self.input_dim)
            word_vec[0,0,self.all_words.index(word)] = 1

            out, hidden = self.lstm(word_vec, hidden)
            
            print(out)
            print(hidden)
            print()
    