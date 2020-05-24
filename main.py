import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from models.basic import mount_basic_model

# Consts
S_WORDS = set(stopwords.words('english'))
VB_SIZE = 5000
EMB_DIM = 64
MAX_LEN = 300
TRUNC_T = 'post'
PADDN_T = 'post'
OOV_TOK = '<OOV>'
FV_CATS = {'SPORTS', 'FOOD & DRINK', 'STYLE & BEAUTY'}
NM_CATS = len(FV_CATS)

# Read dataset using Pandas
df = pd.read_json('./dataset.json')

# Filter by selected categories
df = df[df['category'].isin(FV_CATS)]

# Isolate examples and their related labels
X = df['short_description']
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

X_train_padded = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding=PADDN_T, truncating=TRUNC_T)
print(X_train_padded[10])

# Test(val) to padded seq
X_test_seq = tokenizer.texts_to_sequences(X_test)
print(X_test_seq[10])

X_test_padded = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding=PADDN_T, truncating=TRUNC_T)
print(X_test_padded[10])

# Mount Keras model
model = mount_basic_model(vb_size=VB_SIZE, emb_dim=EMB_DIM, num_classes=NM_CATS)
model.summary()

# Training phase
# TODO(@self): Hyperparameters tuning + charts

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# num_epochs = 10
# history = model.fit(X_train_padded, y_train, epochs=num_epochs,
#                     validation_data=(X_test_padded, y_test), verbose=2)
