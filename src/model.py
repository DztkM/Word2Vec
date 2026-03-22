import numpy as np
import pickle


class SkipGramNS:
    def __init__(self, vocab_size, embedding_dim):
        self.V = vocab_size
        self.D = embedding_dim
        self.W_in = np.random.randn(self.V, self.D) * 0.01
        self.W_out = np.random.randn(self.V, self.D) * 0.01

    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    

    def forward(self, center_idx, context_idx, negatives):
        v_c = self.W_in[center_idx]
        u_o = self.W_out[context_idx]

        score_pos = self.sigmoid(np.dot(u_o, v_c))
        loss = -np.log(score_pos + 1e-10)

        grad_v = (score_pos - 1) * u_o
        grad_u_o = (score_pos - 1) * v_c

        grad_u_neg = []
        for neg in negatives:
            u_k = self.W_out[neg]
            score_neg = self.sigmoid(np.dot(u_k, v_c))

            loss += -np.log(1 - score_neg + 1e-10)

            grad_v += score_neg * u_k
            grad_u_neg.append((neg, score_neg * v_c))

        return loss, grad_v, grad_u_o, grad_u_neg


    def save(self, dataset, path):
        data = {
            "W_in": self.W_in,
            "W_out": self.W_out,
            "word2idx": dataset.word2idx,
            "idx2word": dataset.idx2word,
            "embedding_dim": self.D
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

        print(f"Model saved to {path}")

    
    @classmethod
    def load(cls, path="model.pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)

        vocab_size = len(data["word2idx"])
        embedding_dim = data["embedding_dim"]

        model = cls(vocab_size, embedding_dim)
        model.W_in = data["W_in"]
        model.W_out = data["W_out"]

        print(f"Model loaded from {path}")

        return model, data["word2idx"], data["idx2word"]
