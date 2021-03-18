from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

# text related functions
# reference: https://towardsdatascience.com/how-i-improved-my-text-classification-model-with-feature-engineering-98fbe6c13ef3
def generate_text_features(df, text_col) :
    df['word_count'] = df[text_col].apply(lambda x : len(x.split()))
    df['char_count'] = df[text_col].apply(lambda x : len(x.replace(" ","")))
    df['word_density'] = df['word_count'] / (df['char_count'] + 1)
    df['total_length'] = df[text_col].apply(len)
    df['capitals'] = df[text_col].apply(lambda comment: sum(1 for c in comment if c.isupper()))
    df['capitals_prop'] = df.apply(lambda row: float(row['capitals'])/float(row['total_length']),axis=1)
    df['num_exclamation_marks'] =df[text_col].apply(lambda x: x.count('!'))
    df['num_question_marks'] = df[text_col].apply(lambda x: x.count('?'))
    df['num_punctuation'] = df[text_col].apply(lambda x: sum(x.count(w) for w in '.,;:'))
    df['num_symbols'] = df[text_col].apply(lambda x: sum(x.count(w) for w in '*&$%'))
    df['num_unique_words'] = df[text_col].apply(lambda x: len(set(w for w in x.split())))
    df['prop_unique_words'] = df['num_unique_words'] / df['word_count']
    return df


def process_text(text, lemmatizer, stop_words):
    '''
    This function performs text data preprocessing, including tokenizing the text, converting text to lower case, removing
    punctuation, removing digits, removing stop words, stemming the tokens, then converting the tokens back to strings.
    
    Args:
    ------
        text (string): the text data to be processed
    
    Returns:
    --------
        Returns processed text (string)
    '''
    lemmatizer = WordNetLemmatizer()

    tokens = word_tokenize(text)
    tokens = [w.lower() for w in tokens] #lower case
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens] # remove punctuation
    words = [word for word in stripped if word.isalpha()] # remove non-alphabetic tokens
    words = [w for w in words if not w in stop_words] #remove stopwords
    lemma = [lemmatizer.lemmatize(word) for word in words] #lemmatized 
    processed_text = ' '.join(lemma) #detokenized
    return processed_text