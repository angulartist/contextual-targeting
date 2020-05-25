import re
import string

from nltk.stem import WordNetLemmatizer


def clean_text(text, *, stops=None):
    print(text)
    text = text.translate(string.punctuation)
    text = text.lower()

    text = re.sub(r'(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?', '', text)

    text = re.sub(r'\d+', '', text)

    if stops is not None:
        text = ' '.join([word for word in text.split() if word not in stops])

    text = ' '.join([WordNetLemmatizer().lemmatize(word) for word in text.split() if len(word) > 1])

    return text

# x = clean_text(
#     'Offers a selection of #electronic @parts such as LCD      inverters, u\÷¬ôôô€€€€@}}“—‘ plasma... !! parts, juin 1932 32 - _ ! _°) !§è "bendo" integrated circuits, and fuses and resistors.',
#     stops=set(stopwords.words('english'))
# )
#
# print(x)
