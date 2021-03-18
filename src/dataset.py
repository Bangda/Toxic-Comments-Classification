# Import packages
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import os
import re
from collections import Counter
import random
import pickle
import string
from textblob import TextBlob
from .utils import *

text_column_name = 'comment_text'

if __name__ == "__main__":
    
    # load train data
    data_path = "input/"
    df = pd.read_csv(data_path+"train.csv")
    
    # generate text features
    df = generate_text_features(df, text_column_name)

    # preprocess textP
    lemmatizer=WordNetLemmatizer()
    stop_words=set(stopwords.words('english'))
    df['processed_text'] = df[text_column_name].apply(lambda x: process_text(x, lemmatizer=lemmatizer, stop_words=stop_words))

    # remove null values
    df = df[~df['processed_text'].isnull()]

    # extract polarity and subjectivity
    df.processed_text = df.processed_text.apply(lambda x: str(x))
    df['textblob'] = df['processed_text'].apply(lambda text: TextBlob(text).sentiment)
    df['polarity'] = df['textblob'].apply(lambda x: x[0])
    df['subjectivity'] = df['textblob'].apply(lambda x: x[1])
    df.drop('textblob', 1, inplace=True)

    # export the data
    df.to_csv(data_path +"cleaned_train_from_script.csv", index = False)

