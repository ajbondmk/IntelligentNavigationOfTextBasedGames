import string

class RepresentationGenerator():

    def split_input(self, input_string):
        translator = str.maketrans('', '', string.punctuation)
        sanitised_description = input_string.translate(translator).lower()
        words = sanitised_description.split()
