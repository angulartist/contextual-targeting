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


def mount_conv_model(vb_size, emb_dim, *, num_classes=3, act='relu'):
    """

    :param vb_size:
    :param emb_dim:
    :param num_classes:
    :param act:
    :return:
    """
    return Sequential([
        layers.Embedding(vb_size, emb_dim),
        layers.Dropout(0.2),

        layers.Conv1D(filters=16, kernel_size=3, activation=act),
        layers.GlobalMaxPooling1D(),
        layers.Dense(emb_dim, activation=act),
        layers.Dense(num_classes, activation='softmax'),
    ])
