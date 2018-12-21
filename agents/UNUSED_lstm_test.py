import torch
import torch.nn as nn


input_dim = 10
hidden_dim = 5
lstm = nn.LSTM(input_dim, hidden_dim)

# The axes semantics are (num_layers, minibatch_size, hidden_dim)
hidden = (torch.zeros(1, 1, hidden_dim), torch.zeros(1, 1, hidden_dim))

for i in range(5):
    word_vec = torch.zeros(1,1,input_dim)
    word_vec[0,0,i*2] = 1

    out, hidden = lstm(word_vec, hidden)
    
    print(out)
    print(hidden)
    print()
