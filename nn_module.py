import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """ The neural network model used by Agent02. """


    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_actions):
        """
        Initialise the model: Embedding, then LSTM, then Linear.
            vocab_size = number of possible words
            embedding_dim = word embedding vector size
            hidden_dim = size of LSTM output
            num_actions = number of actions
        """
        super(Model, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden_to_actions = nn.Linear(hidden_dim, num_actions)
        self.hidden_dim = hidden_dim
        self.init_hidden()


    def init_hidden(self):
        """ Initialise the hidden layer to all zeros. """
        self.hidden = (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))


    def forward(self, sentence):
        """ Make a forward pass through the network, returning the final values. """
        embeddings = self.word_embeddings(sentence)
        # TODO: Research following 3 lines.
        lstm_out, self.hidden = self.lstm(
            embeddings.view(len(sentence), 1, -1), self.hidden)
        actions = self.hidden_to_actions(lstm_out.view(len(sentence), -1))
        return actions[-1]
