import numpy as np
cimport numpy as np

cpdef np.ndarray glove_matrix(dict idx, tokenizer, long vb_size, unsigned int emb_dim):
    cdef:
        str word
        unsigned int index
        np.ndarray embedding_matrix
        np.ndarray embedding_vector

    embedding_matrix = np.zeros((vb_size, emb_dim))

    for word, index in tokenizer.word_index.items():
        if index > vb_size - 1:
            break
        else:
            embedding_vector = idx.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector
