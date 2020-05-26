import dill
import numpy as np

from c_libs.core import embedding


def _glove_index(path, dump_path=None):
    """
    Build word/coefs lookup hash-map-huh
    :param path: str Source vec file
    :param dump_path: str Path to save built index
    :return: hash-map-uh
    """
    idx = {}

    with open(path, 'r') as fh:
        for line in fh:
            values = line.split()
            word, *vector = values

            coefs = np.asarray(vector, dtype='float32')
            idx[word] = coefs

    if dump_path is None:
        return idx

    with open(dump_path, 'wb') as fh:
        dill.dump(idx, fh)


def _glove_matrix(path, *, tokenizer, vb_size, emb_dim):
    """
    Build an embedding matrix using GloVe pre-trained weights
    :param path: str Lookup hash-map-uh file path
    :param tokenizer: - Keras trained tokenizer instance
    :param vb_size: int Vocabulary size
    :param emb_dim: int Embedding dimensions
    :return: np.ndarray Embedding matrix used a weights for the embedding layer
    """
    with open(path, 'rb') as fh:
        idx = dill.load(fh)

    embedding_matrix = embedding.glove_matrix(idx, tokenizer, vb_size, emb_dim)

    return embedding_matrix
