import re
import os
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

cur_dir = os.path.dirname(__file__)
stop = pickle.load(open(os.path.join(cur_dir, 'pkl_objects', 'stopwords.pkl'), 'rb'))
english_words = pickle.load(open(os.path.join(cur_dir, 'pkl_objects', 'english_words.pkl'), 'rb'))
vocab = pickle.load(open(os.path.join(cur_dir, 'pkl_objects', 'vocab.pkl'), 'rb'))

def tokenizer(text):
    text = text.decode("utf-8")
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    split_line = text.split()
    split_line = [w for w in split_line if w not in stop]
    split_line = [w for w in split_line if w in english_words]
    split_line = [" ".join(split_line)]
    return np.array(split_line)

vect = CountVectorizer(vocabulary=vocab)
