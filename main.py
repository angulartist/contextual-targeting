import os

import pandas as pd
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from models.basic import mount_conv_model
# Consts
from utils.formatter import clean_text
from utils.pretrained import _glove_matrix

S_WORDS = set(stopwords.words('english'))
VB_SIZE = 10000
EMB_DIM = 50
MAX_LEN = 100
TRUNC_T = 'post'
PADDN_T = 'post'
OOV_TOK = '<OOV>'
FV_CATS = {'Shopping', 'Recreation', 'Sports'}
NM_CATS = len(FV_CATS)

DT_PATH = os.path.join('dataset', 'dmz.csv')
CW_PATH = os.path.join('resources', 'glove.50d.idx.pkl')
CP_PATH = os.path.join('resources', 'w.hdf5')

# Read dataset using Pandas
df = pd.read_csv(DT_PATH, index_col=0)
#
# print(df)
# print(df['category'].unique())
# print(df.groupby(['category']).count())
#
# exit(1)

# Filter by selected categories
df = df[df['category'].isin(FV_CATS)]

# Data cleaning
df = df.dropna()
df = df[df['desc'].apply(lambda e: bool(len(e)))]

print('Cleaning')

# Isolate examples and their related labels
X = (df['desc']
     .apply(lambda e: clean_text(e)))
y = df['category']

print('Samples=', X.shape, 'Labels=', y.shape)

# One-hot encode labels (from str to sparse matrix)
encoder = LabelBinarizer()
y = encoder.fit_transform(y)

# Split dataset to train/test(val)
# TODO(@self): CV
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state=42)

print('Train samples=', X_train.shape, 'Train labels=', y_train.shape)

# Tokenize data
tokenizer = Tokenizer(num_words=VB_SIZE, oov_token=OOV_TOK)
tokenizer.fit_on_texts(X_train)

word_index = tokenizer.word_index
word_items = dict(list(word_index.items())[0:10])
print(word_items)

# Train to padded seq
X_train_seq = tokenizer.texts_to_sequences(X_train)
print(X_train_seq[10])

X_train_padded = pad_sequences(
    X_train_seq,
    maxlen=MAX_LEN,
    padding=PADDN_T,
    truncating=TRUNC_T,
)
print(X_train_padded[10])

# Test(val) to padded seq
X_test_seq = tokenizer.texts_to_sequences(X_test)
print(X_test_seq[10])

X_test_padded = pad_sequences(
    X_test_seq,
    maxlen=MAX_LEN,
    padding=PADDN_T,
    truncating=TRUNC_T,
)
print(X_test_padded[10])

# Pre-trained embedding weights
weights_matrix = _glove_matrix(
    CW_PATH,
    tokenizer=tokenizer,
    vb_size=VB_SIZE,
    emb_dim=EMB_DIM,
)

# Mount Keras model
model = mount_conv_model(
    vb_size=VB_SIZE,
    emb_dim=EMB_DIM,
    max_len=MAX_LEN,
    num_classes=NM_CATS,
    weights=weights_matrix,
)
model.summary()

# Training phase
# TODO(@self): Hyperparameters tuning + charts

checkpoint = ModelCheckpoint(
    CP_PATH,
    save_best_only=True,
    monitor='val_loss',
    mode='min',
)

reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0)

model.compile(
    loss='categorical_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy'],
)

N_EPOCH = 20

history = model.fit(
    X_train_padded,
    y_train,
    validation_data=(X_test_padded, y_test),
    epochs=N_EPOCH,
    batch_size=32,
    callbacks=[checkpoint, reduce_lr_loss],
    verbose=2,
)
