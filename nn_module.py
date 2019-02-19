import torch
import torch.nn as nn


device = torch.device("cuda:0")


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
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, batch_first=True)
        self.linear_1 = nn.Linear(self.hidden_dim, intermediate_dim)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(intermediate_dim, num_actions)
        self.init_weights()


    def init_hidden(self, batch_size):
        """ Initialise the hidden layer to all zeros. """
        self.hidden = (torch.zeros(1, batch_size, self.hidden_dim).to(device),
                torch.zeros(1, batch_size, self.hidden_dim).to(device))
        # TODO: Use torch.nn.init.xavier_uniform


    def init_weights(self):
        for linear in [self.word_embeddings, self.linear_1, self.linear_2]:
            torch.nn.init.xavier_uniform_(linear.weight.data)
            linear.bias.data.fill_(0.0)
        for lstm in [self.lstm]:
            torch.nn.init.xavier_uniform_(lstm.weight_ih_l0)
            torch.nn.init.orthogonal_(lstm.weight_hh_l0)
            for names in lstm._all_weights:
                for name in filter(lambda n: "bias" in n, names):
                    bias = getattr(lstm, name)
                    n = bias.size(0)
                    start, end = n // 4, n // 2
                    bias.data[start:end].fill_(1.0)
            lstm.bias_ih_l0.data.fill_(0.0)


    def forward(self, batch, lengths):
        """ Make a forward pass through the network, returning the final values. """
        embeddings = self.word_embeddings(batch)
        lstm_out, self.hidden = self.lstm(embeddings, self.hidden)
        lstm_out_final = torch.stack([lstm_out[i,lengths[i]-1] for i in range(len(batch))])
        intermediate = self.linear_1(lstm_out_final)
        intermediate = self.relu(intermediate)
        actions = self.linear_2(intermediate)
        return actions
