import dill
import numpy as np
from c_libs.core import embedding


def _glove_index(path, dump_path=None):
    idx = {}

    with open(path, 'r') as fh:
        for line in fh:
            values = line.split()
            word, *rest = values
            print(word, *rest)

            coefs = np.asarray(rest, dtype='float32')
            idx[word] = coefs

    if dump_path is None:
        return idx

    with open(dump_path, 'wb') as fh:
        dill.dump(idx, fh)
        print('Dump index at=', dump_path)


def _glove_matrix(path, *, tokenizer, vb_size, emb_dim):
    with open(path, 'rb') as fh:
        idx = dill.load(fh)

    embedding_matrix = embedding.glove_matrix(idx, tokenizer, vb_size, emb_dim)

    return embedding_matrix
