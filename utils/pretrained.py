import numpy as np


def _glove_index(path):
    idx = {}

    with open(path, 'r') as fh:
        for line in fh:
            values = line.split()
            word, *rest = values
            print(word, *rest)

            coefs = np.asarray(rest, dtype='float32')
            idx[word] = coefs

    return idx


def glove_matrix(path, *, tokenizer, vb_size, emb_dim):
    idx = _glove_index(path)

    embedding_matrix = np.zeros((vb_size, emb_dim))
    for word, index in tokenizer.word_index.items():
        if index > vb_size - 1:
            break
        else:
            embedding_vector = idx.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector
