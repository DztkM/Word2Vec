from dataset import Word2VecDataset
from model import SkipGramNS
from trainer import Trainer
from utils import most_similar, analogy, plot_embeddings_pca


def create_model():
    with open("data/alice_clean.txt", "r") as f:
        text = f.read()

    dataset = Word2VecDataset(
        text, 
        window_size=3,
        max_vocab_size=5000,
        min_freq=3
    )

    model = SkipGramNS(
        vocab_size=dataset.vocab_size,
        embedding_dim=80
    )

    trainer = Trainer(model, dataset, lr=0.05, K=5)
    trainer.train(epochs=50)

    model.save(dataset, "model.pkl")


def load_and_test_model():
    model, word2idx, idx2word = SkipGramNS.load("model.pkl")

    print("\nSimilar to 'cat':", most_similar("cat", model, word2idx, idx2word))
    print("\nSimilar to 'girl':", most_similar("girl", model, word2idx, idx2word))
    print("\nSimilar to 'time':", most_similar("time", model, word2idx, idx2word))
    print("\nSimilar to 'king':", most_similar("king", model, word2idx, idx2word))
    print("\nSimilar to 'queen':", most_similar("queen", model, word2idx, idx2word))
    print("\nSimilar to 'rabbit':", most_similar("rabbit", model, word2idx, idx2word))
    print("\nSimilar to 'tea':", most_similar("tea", model, word2idx, idx2word))
    print("\nSimilar to 'hole':", most_similar("hole", model, word2idx, idx2word))

    
    print(analogy("king", "man", "girl", model, word2idx, idx2word))

    plot_embeddings_pca(model, word2idx, idx2word, num_words=50)



# create_model()
load_and_test_model()