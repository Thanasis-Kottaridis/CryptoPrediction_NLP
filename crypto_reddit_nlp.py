"""
Source Code

nlp on reddit:
https://medium.com/bitgrit-data-science-publication/sentiment-analysis-on-reddit-tech-news-with-python-cbaddb8e9bb6
INSPIRED FROM:
https://www.learndatasci.com/tutorials/sentiment-analysis-reddit-headlines-pythons-nltk/

nlp on reddit with data preprocessing
https://levelup.gitconnected.com/reddit-sentiment-analysis-with-python-c13062b862f6

outher solution:
https://towardsdatascience.com/classifying-reddit-posts-with-natural-language-processing-and-machine-learning-695f9a576ecb

bitcoin NLP paper:
https://www.researchgate.net/publication/352262235_Twitter_and_Reddit_posts_analysis_on_the_subject_of_Cryptocurrencies
http://norma.ncirl.ie/3752/1/Forecasting%20cryptocurrency%20value%20by%20sentiment%20analysis%20chapter.pdf (haven't reed)

bitcoin sentiment analysis API:
https://www.augmento.ai/bitcoin-sentiment/

crypto Bitcoin NLP on reddit posts:
https://cryptomarketpool.com/reddit-sentiment-indicator-for-crypto-in-python/

Reddit WallStreetBets Posts Sentiment Analysis:
https://www.kaggle.com/thomaskonstantin/reddit-wallstreetbets-posts-sentiment-analysis

GENERIC SENTIMENT ANALYSIS FOR CRYPTO ARTICLE:
https://medium.com/general_knowledge/crypto-sentiment-analysis-all-you-need-to-stay-ahead-c8ea69d0f841

"""

# Import Libraries

import pandas as pd
import numpy as np
from pprint import pprint
import time

# visualization imports TODO REMOVE THEM ON DEPLOYED SCRIPT.
import matplotlib.pyplot as plt
from datetime import datetime

# sentiment analysis
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize, RegexpTokenizer  # tokenize words
from nltk.corpus import stopwords

# sentiment preprocessing
import string
import re
import emoji

DATA_FILE_NAME = "Data/reddit_crypto_test_data.csv"  # "reddit_crypto_data.csv"
NLP_DATA_FILE_NAME = 'Data/nlp_crypto_test_data.tsv'


def removePunctuation(text):
    """
        This helper method is used to remove punctuation from a string
        punctuation characters are:
        !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    """
    noPunct = [word for word in text if word not in string.punctuation]
    wordsNoPunct = ''.join(noPunct)
    return wordsNoPunct


def tokenize(text):
    """
    Tokenizing is the process of splitting strings into a list of words.
    We will make use of Regular Expressions or regex to do the splitting.
    Regex can be used to describe a search pattern.

    :param text: string text to be tokenized.
    :return: tokenized text
    """
    split = re.split("\W+", text)
    return split


def removeStopWords(text):
    stopWords = stopwords.words('english')
    text = [word for word in text if word not in stopWords]
    return text


def lemmatizeAndStemming(text):
    # download necessary nltk library.
    nltk.download('omw-1.4')

    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word) for word in text]

    stemmer = PorterStemmer()
    text = [stemmer.stem(word) for word in text]
    return text


def dataPreprocessing(df):
    """
        @TODO Data Cleanning
        @Remove:
        - Check if needed to remove [removed] or [deleted] posts.
        - The post contains “give away” or “giving away”.
        - The post contains “pump”, “register”, or “join”.
        - The post contains more than 14 hashtags
        - The post contains more than 14 ticker symbols.
    """
    # ensure that all titles are Str
    df['title'] = df['title'].astype(str)

    # 1. Remove URLS
    df['clean_title'] = df['title'].apply(lambda x : re.sub(r"http\S+", '', x))

    # 2. remove punctuation from title.
    # df['clean_title'] = df['clean_title'].apply(lambda x: removePunctuation(x))

    # # 3. Remove all the special characters
    # df['clean_title'] = df['clean_title'].apply(lambda x: re.sub(r'\w+', '', x))

    # 3. Remove all emoji
    df['clean_title'] = df['clean_title'].apply(lambda x : emoji.get_emoji_regexp().sub(u'', x))

    # 4. remove all single characters
    df['clean_title'] = df['clean_title'].apply(lambda x: re.sub(r'\s+[a-zA-Z]\s+', '', x))

    # 5. Substituting multiple spaces with single space
    df['clean_title'] = df['clean_title'].apply(lambda x: re.sub(r'\s+', ' ', x))

    # 6. make text to lowercase and tokenize the text
    # df['clean_title'] = df['clean_title'].apply(lambda x: tokenize(x.lower()))

    # 7. remove stop words
    # df['clean_title'] = df['clean_title'].apply(lambda x: removeStopWords(x))

    # 8. Lemmatize / Stem
    # df['clean_title'] = df['clean_title'].apply(lambda x: lemmatizeAndStemming(x))

    # 9. Join tokens in to sentence
    # df['clean_title'] = df['clean_title'].apply(lambda x: " ".join(x))

    # 10. Replace empty titles with NaN
    df['clean_title'].replace(r'^\s*$', np.nan, regex=True, inplace=True)
    df['clean_title'].replace('', np.nan, regex=True, inplace=True)

    # 11. Drop NaN processed titles.
    df = df[df['clean_title'].notna()]

    return df


def getPolarity(df):
    """
    This helper function is used in order to get polarity of all reddit post titles in a given DF.
    polarity is calculated using NLTK and VADER analyzer.

    :param df: dataframe filed with preprocessed reddit posts
    :return: a dataframe that contains posts with their polarity and their label.
    """
    # Downloading NLTK’s databases
    nltk.download('vader_lexicon')  # get lexicons data
    nltk.download('punkt')  # Pre-trained models that help us tokenize sentences.
    nltk.download('stopwords')

    # Initialize Sentiment  Analyzer
    sid = SentimentIntensityAnalyzer()

    # get polarity of each post
    title_res = [*df['clean_title'].apply(sid.polarity_scores)]
    comment_res = [*df['clean_title'].apply(sid.polarity_scores)]
    sentiment_df = pd.DataFrame.from_records(title_res)
    pprint(sentiment_df.head())

    # add polarity columns to DF.
    df = pd.concat([df, sentiment_df], axis=1, join='inner')
    pprint(df.head())

    # Choose labeling threshold
    THRESHOLD = 0.02

    conditions = [
        (df['compound'] <= -THRESHOLD),
        (df['compound'] > -THRESHOLD) & (df['compound'] < THRESHOLD),
        (df['compound'] >= THRESHOLD),
    ]

    # label posts
    values = ["neg", "neu", "pos"]
    df['label'] = np.select(conditions, values)

    return df


if __name__ == '__main__' :
    start_time = time.time()

    # Get Data
    redditPostsDf = pd.read_csv(DATA_FILE_NAME, sep='\t')

    # Data Preprocessing
    redditPostsDf = dataPreprocessing(redditPostsDf)

    # get polarity of all posts
    redditPostsDf = getPolarity(redditPostsDf)

    # plot information about labels
    countStatus = redditPostsDf.label.value_counts()
    pprint(countStatus)
    countStatus.plot.bar(x='Labels', y='Count')
    plt.show()

    # plot compound polarity per day chart
    redditPostsDf = redditPostsDf.copy()
    redditPostsDf['date'] = pd.to_datetime(redditPostsDf['created_iso'], format='%Y-%m-%d %H:%M:%S')
    grouped_df = redditPostsDf.groupby([redditPostsDf['date'].dt.date])['compound'].mean()
    grouped_df.plot.line()
    plt.show()

    # redditPostsDf.plot(x='created_iso', y='compound', style='o')
    #
    # plt.show()

    # write df to TSV
    redditPostsDf.to_csv(NLP_DATA_FILE_NAME, sep='\t', encoding='utf-8')

    print("--- %s seconds ---" % (time.time() - start_time))


