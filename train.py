import numpy as np
import pickle
import argparse


class Train:

    def __init__(self, file_name):
        self.file_name = file_name
        self.necessary_symbols = ['.', '?', '!', ';', '…', ':', ' ']

    def text_normalization(self):
        with open(self.file_name, 'r', encoding='utf8') as file:
            data = file.read().replace('\n', ' ').lower()
        # найдем все буквы
        data_letters = np.unique(list(''.join(filter(str.isalpha, data))))
        # удалим все буквы из текста, определив остальные символы
        letter_dict = dict.fromkeys(data_letters, '')
        table_letters = data.maketrans(letter_dict)
        symbols = np.unique(list(data.translate(table_letters)))
        # уберем . ? ! ; … :
        symbols = list(set(symbols) - set(self.necessary_symbols))
        # удалим все лишние знаки препинания из текста
        symbol_dict = dict.fromkeys(symbols, '')
        table_symbols = data.maketrans(symbol_dict)
        data = data.translate(table_symbols)
        data = data.split(' ')
        indices = np.where(np.array(data) == '')
        data = [i for j, i in enumerate(data) if j not in indices[0]]
        return data

    def fit(self, data):
        text = np.array(data)
        words = np.unique(text)
        words = [word for word in words if word[-1] not in self.necessary_symbols]
        sequences = dict.fromkeys(words)
        text_length = len(text)
        for word in words:
            indices = np.where(text == word)
            if indices[0][-1] == text_length - 1:
                indices = indices[: -1]
            indices = [x + 1 for x in indices]
            next_words = [text[i] for i in indices]
            next_words, amount = np.unique(next_words, return_counts=True)
            next_words = np.array([next_words, amount / np.sum(amount)])
            next_words = next_words.T
            sequences[word] = next_words
        return sequences

    def save_model(self, model, path):
        with open(path,'wb') as f:
            pickle.dump(model, f)


parser = argparse.ArgumentParser()
parser.add_argument('--input-dir', dest='input_dir', type=str)
parser.add_argument('--model', type=str)
args = parser.parse_args()
model = Train(file_name=args.input_dir)
norm_text = model.text_normalization()
fit_model = model.fit(norm_text)
model.save_model(fit_model, args.model)
# python train.py --input-dir path_input --model path_output
