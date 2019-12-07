import unicodedata
import re
import json

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords

def get_ASCII(article):
    '''
    normalizes a string into ASCII characters
    '''

    article = unicodedata.normalize('NFKD', article)\
    .encode('ascii', 'ignore')\
    .decode('utf-8', 'ignore')
    
    return article


def purge_special_characters(article):
    '''
    removes special characters from a string
    '''
    
    article = re.sub(r"[^a-z0-9'\s]", '', article)
    
    return article


def basic_clean(article):
    '''
    calls child functions preforms basic cleaning on a string
    converts string to lowercase, ASCII characters,
    and eliminates special characters
    '''
    # lowercases letters
    article = article.lower()

    # convert to ASCII characters
    article = get_ASCII(article)

    # remove special characters
    article = purge_special_characters(article)
    
    return article


def basic_clean_df(df):
    '''
    takes in a dataframe
    converts text in dataframe to lowercase, ASCII characters
    and eleminates special characters
    '''

    # itterate through columns in dataframe
    for column in df:
    
        # lowercase letters
        df[f'{column}'] = df[f'{column}'].str.lower()

        # convert to ASCII letters
        df[f'{column}'] = df[f'{column}'].apply(get_ASCII)

        # remove special characters
        df[f'{column}'] = df[f'{column}'].apply(purge_special_characters)
        
    return df


def tokenize(article):
    '''
    tokenizes words in a string
    '''

    # create token object
    tokenizer = nltk.tokenize.ToktokTokenizer()

    # use object to tokenize string
    article = tokenizer.tokenize(article, return_str=True)
    
    return article


def stem(article):
    '''
    stems words in a string
    '''

    # create stem object
    ps = nltk.porter.PorterStemmer()
    
    # split article into list of words and stem each word
    stems = [ps.stem(word) for word in article.split()]

    #  join words in list into a string
    article_stemmed = ' '.join(stems)

    return article_stemmed


def lemmatize(article):
    '''
    lemmatizes words in a string
    '''

    # create lemmatize object
    wnl = nltk.stem.WordNetLemmatizer()
    
    # split article into list of words and stem each word
    lemmas = [wnl.lemmatize(word) for word in article.split()]

    #  join words in list into a string
    article_lemmatized = ' '.join(lemmas)
    
    return article_lemmatized


def remove_stopwords(article,extra_words=[],exclude_words=[]):
    '''
    removes stopwords from a string
    user may specify a list of words to add or remove from the list of stopwords
    '''

    # create stopword list using english
    stopword_list = stopwords.words('english')
    
    # remove words in extra_words from stopword list 
    [stopword_list.remove(f'{word}') for word in extra_words]
    
    # add words fin exclude_words to stopword list
    [stopword_list.append(f'{word}') for word in exclude_words]
    
    # slpit article into list of words
    words = article.split()

    # remove words in stopwords from  list of words
    filtered_words = [w for w in words if w not in stopword_list]
    
    # rejoin list of words into article
    article_without_stopwords = ' '.join(filtered_words)
    
    return article_without_stopwords


def prep_article(df):
    '''
    takes in a dataframe with a 'content' column and adds columns 
    showcasing different data cleaning techniques
    '''

    # create column showing content before cleaning
    df['original'] = df.content

    # create column applying basic_cleaning and stem functions
    df['stemmed'] = df.content.apply(basic_clean).apply(stem)

    # create column applying basic_cleaning and lemmatize functions
    df['lemmatized'] = df.content.apply(basic_clean).apply(lemmatize)

    # create column applying basic_cleaning and remove_stopwords functions
    df['clean'] = df.content.apply(basic_clean).apply(remove_stopwords)

    # drop content column
    df = df.drop(columns=['content','Unnamed: 0'])

    
    return df