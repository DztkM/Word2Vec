import numpy as np
import matplotlib.pyplot as plt


def most_similar(word, model, word2idx, idx2word, top_k=5):
    if word not in word2idx:
        return []
    
    v = model.W_in[word2idx[word]]

    sims = model.W_in @ v
    norms = np.linalg.norm(model.W_in, axis=1) * np.linalg.norm(v)
    sims = sims / (norms + 1e-10)

    best = np.argsort(-sims)[:top_k]
    return [idx2word[i] for i in best]


def analogy(a, b, c, model, word2idx, idx2word, top_k=5):
    for w in [a, b, c]:
        if w not in word2idx:
            return []
    
    vec = (model.W_in[word2idx[a]] - model.W_in[word2idx[b]] + model.W_in[word2idx[c]])
    
    sims = model.W_in @ vec
    norms = np.linalg.norm(model.W_in, axis=1) * np.linalg.norm(vec)
    sims = sims / (norms + 1e-10)
    
    best = np.argsort(-sims)[:top_k]
    return [idx2word[i] for i in best]


def plot_embeddings_pca(model, word2idx, idx2word, num_words=50):
    W = model.W_in

    words = list(word2idx.keys())[:num_words]
    
    indices = [word2idx[w] for w in words]

    X = W[indices]
    X_mean = X - np.mean(X, axis=0)
    U, S, Vt = np.linalg.svd(X_mean)
    X_2d = U[:, :2] @ np.diag(S[:2])

    plt.figure()
    plt.scatter(X_2d[:, 0], X_2d[:, 1])

    for i, word in enumerate(words):
        plt.annotate(word, (X_2d[i, 0], X_2d[i, 1]))

    plt.title("Word Embeddings (PCA)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.show()