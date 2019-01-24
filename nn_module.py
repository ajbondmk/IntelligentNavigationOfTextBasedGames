import torch
import torch.nn as nn


class Model(nn.Module):
    """ The neural network model used by Agent02. """


    def __init__(self, vocab_size, num_words, num_actions):
        """
        Initialise the model: Embedding, then LSTM, then Linear.
            vocab_size = number of possible words
            num_words = number of words in each input
            num_actions = number of actions
        """
        
        super(Model, self).__init__()

        embedding_dim = 20
        self.hidden_dim = 100
        intermediate_dim = 64

        self.word_embeddings = nn.Linear(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim)
        self.linear_1 = nn.Linear(self.hidden_dim, intermediate_dim)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(intermediate_dim, num_actions)


    def init_hidden(self, batch_size):
        """ Initialise the hidden layer to all zeros. """
        self.hidden = (torch.zeros(1, batch_size, self.hidden_dim),
                torch.zeros(1, batch_size, self.hidden_dim))
        # TODO: Use torch.nn.init.xavier_uniform


    def forward(self, batch):
        """ Make a forward pass through the network, returning the final values. """
        embeddings = self.word_embeddings(batch)
        lstm_out, self.hidden = self.lstm(embeddings.view(50,len(batch),-1), self.hidden)
        intermediate = self.linear_1(lstm_out.view(len(batch),50,-1))
        intermediate = self.relu(intermediate)
        actions = self.linear_2(intermediate)
        final_actions = torch.stack([actions[i,-1] for i in range(len(actions))])
        return final_actions
