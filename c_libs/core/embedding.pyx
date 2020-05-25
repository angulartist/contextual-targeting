import numpy as np

cpdef glove_matrix(dict idx, tokenizer, long vb_size, unsigned int emb_dim):
    cdef:
        str word
        unsigned int index

    embedding_matrix = np.zeros((vb_size, emb_dim))

    for word, index in tokenizer.word_index.items():
        if index > vb_size - 1:
            break
        else:
            embedding_vector = idx.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector
