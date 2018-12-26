import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):

    # vocab_size = number of possible words
    # embedding_dim = word embedding vector size
    # hidden_dim = size of LSTM output
    # num_commands = number of commands
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_commands):
        super(Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden_to_commands = nn.Linear(hidden_dim, num_commands)
        self.init_hidden()

    def init_hidden(self):
        self.hidden = (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, sentence):
        embeddings = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(
            embeddings.view(len(sentence), 1, -1), self.hidden)
        commands = self.hidden_to_commands(lstm_out.view(len(sentence), -1))
        # commands2 = F.log_softmax(commands, dim=1)
        return commands
