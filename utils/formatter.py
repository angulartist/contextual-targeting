import re


def preprocess_text(seq):
    """
    Quick and dirty text cleaning.
    :param seq: str Text sequence to clean.
    :return: str Cleaned sequence.
    """
    seq = re.sub('[^a-zA-Z]', ' ', seq)
    seq = re.sub(r"\s+[a-zA-Z]\s+", ' ', seq)
    seq = re.sub(r'\s+', ' ', seq)

    return seq
