# utils/tokenizer.py

import pickle
from collections import defaultdict

class SimpleTokenizer:
    def __init__(self):
        self.word2idx = defaultdict(lambda: 0)
        self.idx2word = {}
        self.next_index = 1

    def fit(self, texts):
        for text in texts:
            for word in text.split():
                if word not in self.word2idx:
                    self.word2idx[word] = self.next_index
                    self.idx2word[self.next_index] = word
                    self.next_index += 1

    def encode(self, text):
        return [self.word2idx[word] for word in text.split()]

    def decode(self, indices):
        return ' '.join(self.idx2word.get(i, '?') for i in indices)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump((dict(self.word2idx), self.idx2word), f)

    def load(self, path):
        with open(path, 'rb') as f:
            w2i, i2w = pickle.load(f)
            self.word2idx = defaultdict(lambda: 0, w2i)
            self.idx2word = i2w
