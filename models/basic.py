from keras import Sequential, layers


def mount_basic_model(vb_size, emb_dim, *, num_classes=3, act='relu'):
    """
    :param vb_size:
    :param emb_dim:
    :param num_classes:
    :param act:
    :return:
    """
    return Sequential([
        layers.Embedding(vb_size, emb_dim),
        layers.Bidirectional(layers.LSTM(emb_dim, recurrent_dropout=0.2)),
        layers.Dense(emb_dim, activation=act),
        layers.Dense(num_classes, activation='softmax'),
    ])


def mount_conv_model(vb_size, emb_dim, *, max_len=100, num_classes=3, act='relu', weights=None):
    """

    :param max_len:
    :param weights:
    :param vb_size:
    :param emb_dim:
    :param num_classes:
    :param act:
    :return:
    """
    model = Sequential([
        layers.Embedding(vb_size, emb_dim, input_length=max_len),
        layers.Conv1D(filters=16, kernel_size=3, activation=act),
        layers.MaxPooling1D(pool_size=4),
        layers.Flatten(),
        layers.Dense(32, activation=act),
        layers.Dense(num_classes, activation='softmax'),
    ])

    # If any pre-trained weights (glove/w2v), use em
    if weights is not None:
        model.layers[0].set_weights([weights])
        model.layers[0].trainable = False

    return model
