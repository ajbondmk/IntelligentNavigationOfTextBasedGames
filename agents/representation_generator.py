import string
import torch
import torch.nn as nn

# TODO: Add description.
class RepresentationGenerator():

    def __init__(self):
        f = open('all_words.txt', 'r')
        self.all_words = f.read().split('\n')
        f.close()

    def split_input(self, input_string):
        translator = str.maketrans('', '', string.punctuation)
        sanitised_description = input_string.translate(translator).lower()
        words = sanitised_description.split()

        for word in words:
            word_vec = torch.zeros(len(self.all_words))
            word_vec[self.all_words.index(word)] = 1
