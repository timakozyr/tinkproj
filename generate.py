import numpy as np
import pickle
import argparse


class Generator:

    def __init__(self, length):
        self.length = length
        self.necessary_symbols = ['.', '?', '!', ';', '…', ':', ' ']

    def generate(self, sequences):
        words = list(sequences.keys())
        first_word = np.random.choice(words)
        self.length = self.length - 1
        sentence = [first_word]
        prev_word = first_word
        while self.length > 0:
            if prev_word not in list(sequences.keys()):
                prev_word = np.random.choice(list(sequences.keys()))
            max_val = max(sequences[prev_word].T[1])
            indices = [i for i, x in enumerate(sequences[prev_word].T[1]) if x == max_val]
            idx = np.random.choice(indices)
            next_word = sequences[prev_word].T[0][idx]
            while next_word[-1] in self.necessary_symbols:
                next_word = next_word[: -1]
            sentence.append(next_word)
            prev_word = next_word
            self.length = self.length - 1
        sentence = ' '.join(sentence)
        return sentence.capitalize()

    def load(self, model):
        with open(model, 'rb') as f:
            fit_model = pickle.load(f)
            return fit_model


parser = argparse.ArgumentParser()
parser.add_argument('--length', type=int)
parser.add_argument('--model', type=str)
args = parser.parse_args()
model = args.model
generator = Generator(args.length)
fit_model = generator.load(model)
sentence = generator.generate(fit_model)
print(sentence)
