import torch
import torch.nn as nn
import torch.nn.functional as F
import sys


class Model(nn.Module):
    """ The neural network model used by Agent02. """


    def __init__(self, vocab_size, num_words, embedding_dim, hidden_dim, num_actions):
        """
        Initialise the model: Embedding, then LSTM, then Linear.
            vocab_size = number of possible words
            num_words = number of words in each input
            embedding_dim = word embedding vector size
            hidden_dim = size of LSTM output
            num_actions = number of actions
        """
        
        super(Model, self).__init__()

        self.num_words = num_words
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Linear(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden_to_actions = nn.Linear(hidden_dim, num_actions)

        self.init_hidden()


    def init_hidden(self):
        """ Initialise the hidden layer to all zeros. """
        self.hidden = (torch.zeros(1, self.num_words, self.hidden_dim),
                torch.zeros(1, self.num_words, self.hidden_dim))


    def forward(self, batch):
        """ Make a forward pass through the network, returning the final values. """
        embeddings = self.word_embeddings(batch)
        lstm_out, self.hidden = self.lstm(embeddings, self.hidden)
        actions = self.hidden_to_actions(lstm_out)
        final_actions = torch.stack([actions[i,-1] for i in range(len(actions))])
        return final_actions
