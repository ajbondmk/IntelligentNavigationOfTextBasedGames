import string

class RepresentationGenerator():

    def get_all_words(self):
        f = open('all_words.txt', 'r')
        self.all_words = f.read().split('\n')
        f.close()

    def split_input(self, input_string):
        translator = str.maketrans('', '', string.punctuation)
        sanitised_description = input_string.translate(translator).lower()
        words = sanitised_description.split()
