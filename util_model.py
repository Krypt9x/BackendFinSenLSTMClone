from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from bs4 import BeautifulSoup
import numpy as np
import re
import nltk


def cleanText(text):
    text = BeautifulSoup(text, "lxml").text
    text = re.sub(r'\|\|\|', r' ', text)
    text = re.sub(r'http\S+', r'<URL>', text)
    text = text.lower()
    text = text.replace('x', '')
    return text

def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) <= 0:
                continue
            tokens.append(word.lower())
    return tokens

def print_pred_sentiment(msg, tokenizer, model, X):
  seq = tokenizer.texts_to_sequences(msg)
  padded = pad_sequences(seq, maxlen=X.shape[1], dtype='int32', value=0)
  pred = model.predict(padded)
  labels = ['0','1','2']
  index_max = np.argmax(pred)
  if index_max == 0:
    return "Positif"
  elif index_max == 1:
    return "Netral"
  elif index_max == 2:
    return "Negatif"