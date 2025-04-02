import time
import json
import string
import random
import pickle
import pathlib

import pandas as pd

from nltk import corpus
from datetime import datetime


def get_system_time():
    return str(time.time()).split(".")[0]


def get_current_datetime():
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def generate_random_string(k):
    return "".join(random.sample(string.punctuation + string.ascii_letters + " ", k))


def generate_random_sentence():
    lst = [" ".join(tokens) for tokens in corpus.gutenberg.sents('shakespeare-macbeth.txt') if len(tokens) >= 10]
    return random.choice(lst)


def set_pandas_display(max_colwidth=100):
    pd.options.display.max_rows = None
    pd.options.display.max_colwidth = max_colwidth
    pd.options.display.max_columns = None


def load_json_file(filename):
    with open(filename, "r") as fp:
        data = json.load(fp)

    return data


def save_pickle_file(data, file_name):
    with open(file_name, "wb") as fp:
        pickle.dump(data, fp)


def load_pickle_file(file_name):
    with open(file_name, "rb") as fp:
        return pickle.load(fp)
