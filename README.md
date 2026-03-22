# Word2Vec implementation in pure NumPy

A clean, minimal implementation of the Word2Vec algorithm using pure NumPy, without any deep learning frameworks. This project focuses on understanding and reproducing the core mechanics of training word embeddings from scratch.


## Features
- Skip-gram model with Negative Sampling
- Full training loop implemented manually:
  + forward pass
  + loss computation
  + gradient calculation
  + parameter updates
- Subsampling of frequent words (improves embedding quality)
- Model saving and loading


## Results

### With subsampling (Epoch 50, Loss: 148546.0774)
```
Similar to 'cat': ['cat', 'cheshire', 'dog', 'love', 'allow']
Similar to 'girl': ['girl', 'serpent', 'crab', 'something', 'knowledge']
Similar to 'time': ['time', 'lost', 'unfortunate', 'paper', 'away']
Similar to 'king': ['king', 'staring', 'myself', 'wrote', 'verdict']
Similar to 'queen': ['queen', 'players', 'staring', 'em', 'taken']
Similar to 'rabbit': ['rabbit', 'white', 'name', 'hush', 'kid']
Similar to 'tea': ['tea', 'bread', 'pointing', 'telescope', 'week']
Similar to 'hole': ['hole', 'passage', 'grunted', 'jumping', 'bat']

rabbit - white + cat': ['cat', 'cheshire', 'dog', 'never', 'name']
```

### Without subsampling

```
Similar to 'cat': ['cat', 'at', 'cheshire', 'judge', 'letter']
Similar to 'girl': ['girl', 'show', 'telescope', 'eaglet', 'serpent']
Similar to 'time': ['time', 'silence', 'minutes', 'difficulty', 'notion']
Similar to 'king': ['king', 'outside', 'wrote', 'guessed', 'somewhere']
Similar to 'queen': ['queen', 'staring', 'hoarse', 'procession', 'fear']
Similar to 'rabbit': ['rabbit', 'white', 'hush', 'noticed', 'kid']
Similar to 'tea': ['tea', 'conversation', 'ears', 'bread', 'a']
Similar to 'hole': ['hole', 'passage', 'grunted', 'white', 'a']

rabbit - white + cat': ['cat', 'our', 'usual', 'first', 'head']
```

### Key Insight

Subsampling frequent words significantly improves the quality of learned embeddings by reducing noise from overly common tokens (e.g., "the", "a"). This leads to more meaningful semantic relationships and better analogy performance.

### Data Source
[Alice’s Adventures in Wonderland - PROJECT GUTENBERG](https://www.gutenberg.org/files/11/11-h/11-h.htm)


## Project Structure
```
.
├── src/
│   ├── word2vec.py         # entry point: runs training, evaluation, and experiments
│   ├── model.py            # Skip-gram model and embedding parameters
│   ├── utils.py            # utility functions (sampling, similarity, etc.)
│   ├── dataset.py          # dataset preparation and batching logic
│   ├── text_cleaner.py     # text preprocessing and normalization pipeline
│   └── trainer.py          # training loop, loss computation, and optimization
├── data/
│   ├── alice_raw.txt       # original raw text (Project Gutenberg)
│   └── alice_clean.txt     # preprocessed training corpus
└── README.md
```

## How to run

1. Install dependencies
    ```bash
    uv sync
    ```

2. Run training
    ```
    uv run .\src\word2vec.py
    ```
3. Notes
    - Make sure the dataset is located in the `data/` directory
    - You can modify training parameters (epochs, window size, embedding dim, etc.) directly in `word2vec.py`