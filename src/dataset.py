import numpy as np
from collections import Counter


class Word2VecDataset:
    def __init__(self, text, window_size=2, max_vocab_size=None, min_freq=1):
        self.window_size = window_size
        
        tokens = text.lower().split()
        counter = Counter(tokens)
        tokens = [t for t in tokens if counter[t] >= min_freq]

        if max_vocab_size is not None:
            most_common = set(
                [w for w, _ in counter.most_common(max_vocab_size)]
            )
            tokens = [t for t in tokens if t in most_common]

        self.tokens = tokens
        

        self.vocab = list(set(self.tokens))
        self.word2idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self.vocab_size = len(self.vocab)

        self.pairs = self._generate_pairs()


    def _generate_pairs(self):
        pairs = []
        for i, word in enumerate(self.tokens):
            center = self.word2idx[word]
            for j in range(-self.window_size, self.window_size + 1):
                if j == 0 or i + j < 0 or i + j >= len(self.tokens):
                    continue
                context = self.word2idx[self.tokens[i + j]]
                pairs.append((center, context))
        
        return pairs
    

    def get_negative_samples(self, pos_idx, K):
        negatives = []
        while len(negatives) < K:
            neg = np.random.randint(0, self.vocab_size)
            # TODO maybe should check whole context, not just pos_idx
            if neg != pos_idx: 
                negatives.append(neg)

        return negatives
    