from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import nltk
import util_model

DATASET_PATH = './resource/all-data.csv'
MODEL_PATH = './resource/FinancialSentimentModel.keras'
max_fatures = 500000


class InitModel:
    def __init__(self) -> None:
        nltk.download('punkt_tab')
        self.df = self.read_csv()
        self.init_df()
        self.loaded_model = self.init_model()
        self.loaded_tokenizer = self.init_tokenizer(self.df)

    def read_csv(self):
        df = pd.read_csv(DATASET_PATH, encoding='latin-1')
        print(f'Membaca csv...\n')
        return df
    
    def init_df(self):
        self.df = self.df.rename(columns={'neutral':'sentiment','According to Gran , the company has no plans to move all production to Russia , although that is where the company is growing .':'Message'})
        self.df.index = range(4845)
        self.df['Message'].apply(lambda x: len(x.split(' '))).sum()
        sentiment  = {'positive': 0,'neutral': 1,'negative':2}
        self.df.sentiment = [sentiment[item] for item in self.df.sentiment]
        print(f'Df siap...\n')

    def init_model(self):
        loaded_model = load_model(MODEL_PATH)
        return loaded_model
    
    def init_tokenizer(self, df):
        df['Message'] = df['Message'].apply(util_model.cleanText)
        tokenizer = Tokenizer(num_words=max_fatures, split=' ', filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
        tokenizer.fit_on_texts(df['Message'].values)
        self.X = tokenizer.texts_to_sequences(df['Message'].values)
        self.X = pad_sequences(self.X)
        print('Found %s unique tokens.' % len(self.X))
        return tokenizer

