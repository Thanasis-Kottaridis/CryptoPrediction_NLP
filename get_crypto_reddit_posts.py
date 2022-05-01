"""
Documentation sites

Scrapping Reddit data
https://towardsdatascience.com/scraping-reddit-data-1c0af3040768

How to Use the Reddit API in Python
https://towardsdatascience.com/how-to-use-the-reddit-api-in-python-5e05ddfd1e5c

Reddit API
https://www.reddit.com/dev/api#GET_search

psaw DOC
https://psaw.readthedocs.io/en/latest/
https://pushshift.io/api-parameters/

PRAW DOC
https://praw.readthedocs.io/en/latest/code_overview/models/subreddit.html

Best crypto subreddits
https://coinbound.io/best-crypto-subreddits/

"""

# Get from
import pymongo
import utils
import praw
from psaw import PushshiftAPI
import pandas as pd
import time
import crypto_reddit_nlp as crn

# Store Type
# True => Store in monge
# False => Store in CSV.
shouldStoreInMongo = True

# Target Timestamp
# Friday, October 29, 2021 5:07:29 PM
TARGET_TIMESTAMP = 1635527249

# data file name:
file_name = "reddit_crypto_test_data.csv"  # "reddit_crypto_data.csv"

# REDDIT AUTH
CLIENT_ID = "IcTrWsQDFCcZEe3rWrlB4A"

SECRET_KEY = 'HZQy-nneDNv4THu_G8MhVJ96KOq4cg'

if __name__ == '__main__' :

    # get mongo client
    client, db = utils.connectMongoDB()
    collection = db.reddit_crypto_data

    # connect to reddit api
    reddit = praw.Reddit(
        client_id=CLIENT_ID,
        client_secret=SECRET_KEY,
        user_agent='MyBot/0.0.1'
    )

    api = PushshiftAPI(reddit)

    print("-------- CURRENT UNIX --------")
    print(f"{time.time()}")
    print("--------------------------------")

    FROM_TIMESTAMP = int(time.time())
    if shouldStoreInMongo:
        """
            @SOS!!!!
            Change target timestamp to most recent post stored in MONGO.
            If documents exists in mongo that's mean that this script has run again with flag shouldStoreInMongo = True
            so we need to insert to mongo only reddit posts performed after last run.
        """
        cursor = collection.find().sort("created_unix", pymongo.DESCENDING).limit(1)
        for doc in cursor:
            TARGET_TIMESTAMP = doc["created_unix"]

    # Use This if you want to write in .CSV
    # initialize dataframe
    df = pd.DataFrame()

    while True :

        # Use this if you want to store it in mongo.
        if shouldStoreInMongo :
            # initialize dataframe
            df = pd.DataFrame()

        # The `search_comments` and `search_submissions` methods return generator objects
        try:
            gen = api.search_submissions(
                before=int(FROM_TIMESTAMP),
                after=int(TARGET_TIMESTAMP),
                subreddit="Bitcoin",
                sort="desc",
                limit=100
            )
        except:
            print("Empty response no more data")
            break

        # check if response has data
        results = list(gen)
        if len(results) == 0:
            print("Empty response no more data")
            break

        for post in results :
            # append relevant data to dataframe

            df = df.append({
                'id' : post.id,
                'subreddit' : str(post.subreddit),
                'fullname' : post.name,
                'title' : post.title,
                'selftext' : post.selftext,
                'upvote_ratio' : post.upvote_ratio,
                'ups' : post.ups,
                'downs' : post.downs,
                'score' : post.score,
                'created_iso' : utils.utc_to_datetime(post.created),
                'created_unix' : post.created
            }, ignore_index=True)

        # Preform NLP Analysis for reddit batch.
        # - First perform preprocessing
        # - And then NLP using VENDER.
        df = crn.dataPreprocessing(df)
        df = crn.getPolarity(df)

        # get last subreddit created time
        created_unix = int(df.iloc[-1 :].created_unix)

        print("-----------------------------------------------")
        print(f"POSTS COUNT: {len(df.index)}")
        print(f"created_unix: {created_unix}, TARGET_TIMESTAMP: {TARGET_TIMESTAMP}")
        print("-----------------------------------------------")

        if created_unix <= TARGET_TIMESTAMP :
            print(f"created_unix <= TARGET_TIMESTAMP: True")
            break
        else :
            FROM_TIMESTAMP = created_unix

        if shouldStoreInMongo :
            # write to Mongo
            print("Insert to Mongo")
            collection.insert_many(df.to_dict('records'))
        else:
            # write df to CSV
            print("Store DF")
            df.to_csv(file_name, sep='\t', encoding='utf-8')
