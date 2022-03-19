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
import matplotlib.pyplot as plt
from datetime import datetime

# sentiment analysis
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, RegexpTokenizer  # tokenize words
from nltk.corpus import stopwords

DATA_FILE_NAME = "reddit_crypto_data.csv"
NLP_DATA_FILE_NAME = 'nlp_crypto_data.tsv'

if __name__ == '__main__' :
    # Downloading NLTK’s databases
    nltk.download('vader_lexicon')  # get lexicons data
    nltk.download('punkt')  # Pre-trained models that help us tokenize sentences.
    nltk.download('stopwords')

    # Initialize Sentiment  Analyzer
    sid = SentimentIntensityAnalyzer()

    # Get Data
    redditPostsDf = pd.read_csv(DATA_FILE_NAME, sep='\t')

    """
        @TODO Data Cleanning
        @Remove:
        - The post contains “give away” or “giving away”.
        - The post contains “pump”, “register”, or “join”.
        - The post contains more than 14 hashtags
        - The post contains more than 14 ticker symbols.
    """


    # get polarity of each post
    title_res = [*redditPostsDf['title'].apply(sid.polarity_scores)]
    comment_res = [*redditPostsDf['title'].apply(sid.polarity_scores)]
    sentiment_df = pd.DataFrame.from_records(title_res)
    pprint(sentiment_df.head())

    # add polarity columns to DF.
    redditPostsDf = pd.concat([redditPostsDf, sentiment_df], axis=1, join='inner')
    pprint(redditPostsDf.head())

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


