import string
import torch
import torch.nn as nn


# TODO: Add description.
class RepresentationGenerator():


    def __init__(self):
        
        self.commands = ['go north', 'go east', 'go south', 'go west', 'take coin']

        # Read all possible words into an array.
        f = open('word_lists/all_words.txt', 'r')
        all_words = f.read().split('\n')
        f.close()
        
        # Create LSTM module.
        self.input_dim = len(all_words)
        self.hidden_dim = len(self.commands)
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim)
        
        # Create dictionary of all words to one-hot encoded vectors.
        self.word_to_vec = {}
        for i in range(self.input_dim):
            vec = torch.zeros(1,1,self.input_dim)
            vec[0,0,i] = 1
            self.word_to_vec[all_words[i]] = vec


    # TODO: Rename.
    def select_action(self, input_string):

        # Split the input description text into lowercase words with no punctuation.
        translator = str.maketrans('', '', string.punctuation)
        sanitised_description = input_string.translate(translator).lower()
        input_words = sanitised_description.split()

        # Remove all stop words and unknown words.
        words = []
        for input_word in input_words:
            if input_word in self.word_to_vec:
                words.append(input_word)
        
        # Create the input vector.
        num_words = len(words)
        input_vec = torch.zeros(1, num_words, len(self.word_to_vec))
        for i in range(num_words):
            print([words[i]])
            input_vec[0,i] = self.word_to_vec[words[i]]
        
        # Initialise the hidden state.
        # TEMP: The axes semantics are (num_layers, minibatch_size, hidden_dim).
        # TEMP: Alternatively could use `torch.randn`.
        hidden = (torch.zeros(1, num_words, self.hidden_dim), torch.zeros(1, num_words, self.hidden_dim))

        # Pass the input vector to the LSTM module.
        out, hidden = self.lstm(input_vec, hidden)
        print(out[0,num_words-1])
        _,b = torch.max(out[0,num_words-1],0)
        command = self.commands[b]
        print(command)
        return command


# representation_generator = RepresentationGenerator()
# representation_generator.select_action("this is a vault of rooms")
