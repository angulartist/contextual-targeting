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
        layers.Conv1D(32, 7, activation=act),
        layers.BatchNormalization(),
        layers.MaxPool1D(5),
        layers.Dropout(0.2),

        layers.Conv1D(32, 7, activation=act),
        layers.BatchNormalization(),
        layers.GlobalMaxPooling1D(),
        layers.Dense(emb_dim, activation=act),
        layers.Dense(num_classes, activation='softmax'),
    ])
