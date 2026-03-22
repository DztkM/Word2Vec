import random


class Trainer:
    def __init__(self, model, dataset, lr=0.05, K=5):
        self.model = model
        self.dataset = dataset
        self.lr = lr
        self.K = K

    
    def train(self, epochs=20):
        pairs = self.dataset.pairs

        for epoch in range(epochs):
            random.shuffle(pairs)
            total_loss = 0

            for center, context in pairs:
                negatives = self.dataset.get_negative_samples(context, self.K)
                loss, grad_v, grad_u_o, grad_u_neg = self.model.forward(center, context, negatives)

                self.model.W_in[center] -= self.lr * grad_v
                self.model.W_out[context] -= self.lr * grad_u_o
                for neg_idx, grad in grad_u_neg:
                    self.model.W_out[neg_idx] -= self.lr * grad

                total_loss += loss

            print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
    