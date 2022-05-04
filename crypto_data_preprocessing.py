import utils
import numpy as np
import pandas as pd
import seaborn as sns
from pprint import pprint
import matplotlib.pyplot as plt
import json
from bson.json_util import loads, dumps

# Consts:
shouldUseMongoData = True  # If true fetch data from processed_crypto_data Collection Else fetch from local CSV
shouldCreateCSV = True  # if true store flatten data to CSV file
doPlots = True
file_name = "flatten_crypto_data.csv"

# Unix Time Consts
ONE_MINUTE_SECONDS = 60  # 60 seconds


def cleanMongoData(storeToMongo=False) :
    """
    This helper fun is used to clean all crypto data selected from bitcoin API
    and stored into processed_crypto_data collection.

    This functions selects only the most valuable fields and inserts them in to a new
    mongo Collection named flatten_crypto_data

    :return: It returns flatten_data dictionary
    and a separate list with ids (ids, flatten_data)
    """
    target_elements = [
        "OHLC_close",  # OHLC close last 1 minute
        "OHLC_high",  # OHLC high last 1 minute
        "OHLC_low",  # OHLC low last 1 minute
        "OHLC_open",  # OHLC open last 1 minute
        # "OHLC_timestamp",
        # "OHLC_timestamp_iso",
        "OHLC_volume",  # OHLC volume last 1 minute
        "Quotes_BTC_circulating_supply",
        # "Quotes_BTC_cmc_rank", # not need this
        # "Quotes_BTC_date_added", # not need this
        # "Quotes_BTC_id", # not need this
        # "Quotes_BTC_is_active", # not need this.
        # "Quotes_BTC_is_fiat", # not need this
        # "Quotes_BTC_last_updated", # not need this
        "Quotes_BTC_max_supply",
        # "Quotes_BTC_name", # not need this
        # "Quotes_BTC_num_market_pairs", # ????
        # "Quotes_BTC_platform", # not need this,
        "Quotes_BTC_quote_USD_fully_diluted_market_cap",
        # "Quotes_BTC_quote_USD_last_updated", # Keep one timestamp in Unix format
        "Quotes_BTC_quote_USD_market_cap",
        "Quotes_BTC_quote_USD_market_cap_dominance",
        "Quotes_BTC_quote_USD_percent_change_1h",
        "Quotes_BTC_quote_USD_percent_change_24h",
        "Quotes_BTC_quote_USD_percent_change_30d",
        "Quotes_BTC_quote_USD_percent_change_60d",
        "Quotes_BTC_quote_USD_percent_change_7d",
        "Quotes_BTC_quote_USD_percent_change_90d",
        "Quotes_BTC_quote_USD_price",
        "Quotes_BTC_quote_USD_volume_24h",
        "Quotes_BTC_quote_USD_volume_change_24h",
        # "Quotes_BTC_tags", # not need.
        "Quotes_BTC_total_supply",
        "Ticker_ask",  # OHLC ask last 24h
        "Ticker_bid",  # OHLC bid last 24h
        "Ticker_high",  # OHLC high last 24h
        "Ticker_last",  # OHLC last last 24h
        "Ticker_low",  # OHLC low last 24h
        "Ticker_open",  # OHLC open last 24h
        "Ticker_timestamp",  # keep one timestamp
        # "Ticker_timestamp_iso",  # Keep one timestamp in Unix format
        "Ticker_volume",  # OHLC volume last 24h
        "Ticker_vwap",  # OHLC vwap last 24h
    ]

    # connecting or switching to the database
    connection, db = utils.connectMongoDB()

    # creating or switching to ais_navigation collection
    collection = db.processed_crypto_data

    # Mongo response
    res = collection.find()

    jsonData = list(res)

    # create flatten data.
    flatten_data = []
    print(len(jsonData))
    for doc in jsonData :
        flatten_data.append(utils.flatten_json(doc, target_elements))

    ids = None
    if storeToMongo :
        ids = storeFlattenData(flatten_data)

    return ids, flatten_data


def cleanOutliers(df, columnList) :
    # Check Each column for outliers.
    df_filtered = df.copy()
    for col in columnList :
        # quantile column
        q_low = df_filtered[str(col)].quantile(0.03)
        q_hi = df_filtered[str(col)].quantile(0.97)

        mask = ((df_filtered[str(col)] > q_hi) | (df_filtered[str(col)] < q_low))
        df_filtered.loc[mask, str(col)] = np.nan

    # check for null values per column after remove outliers
    print("NaN values per column count: \n")
    print(df_filtered.isna().sum())

    # Use interpolate to fill NaN values created by removing outliers
    df_filtered = df_filtered.interpolate(limit_direction="both")

    return df_filtered


def storeFlattenData(flatten_data) :
    # connecting or switching to the database
    connection, db = utils.connectMongoDB()

    clean_collection = db.flatten_crypto_data
    # drop collection if exists
    clean_collection.drop()
    # insert new documents
    ids = clean_collection.insert_many(flatten_data)
    return ids


def getRedditPostsFromMongo():
    # connecting or switching to the database
    connection, db = utils.connectMongoDB()

    # creating or switching to ais_navigation collection
    collection = db.reddit_crypto_data

    # get reddit posts from mongo
    res = collection.find()
    jsonData = list(res)

    # create Dataframe with results
    resultsDf = pd.DataFrame().from_dict(jsonData)

    return resultsDf


def getWeightedRedditPolarity(df, timeFrom, timeRange) :

    # calculate timeTo
    timeTo = timeFrom + timeRange

    # filter results df
    resultsDf = df[(df["created_unix"] < timeTo) & (df["created_unix"] >= timeFrom)]

    scoreSum = resultsDf["score"].sum()
    print("-------- weighted_polarity calculated --------")
    if scoreSum == 0:
        res = resultsDf['weighted_polarity'].sum()
        print(f"-------- {res} -------- \n")
        return res
    else:
        res = resultsDf['weighted_polarity'].sum() / resultsDf["score"].sum()
        print(f"-------- {res} -------- \n")
        return res


def grouped_weighted_avg(values, weights) :
    if weights == 0:
        return 0
    else:
        return (values * weights) / weights


if __name__ == '__main__' :

    if shouldUseMongoData :
        # clean data from mongo
        # and keep only features needed for prediction
        _, flattenData = cleanMongoData()

        # Create Df with flatten values
        flatten_df = pd.DataFrame.from_dict(flattenData)
        # ensure that data are numeric
        flatten_df = flatten_df.apply(pd.to_numeric)

        # Column Rename on Flatten DF
        flatten_df.rename(
            columns={
                "OHLC_close" : "close_1min",
                "OHLC_high" : "high_1min",
                "OHLC_low" : "low_1min",
                "OHLC_open" : "open_1min",
                "OHLC_volume" : "volume_1min",
                "Quotes_BTC_max_supply" : "max_supply",
                "Quotes_BTC_quote_USD_fully_diluted_market_cap" : "fully_diluted_market_cap",
                "Quotes_BTC_circulating_supply" : "circulating_supply",
                "Quotes_BTC_quote_USD_market_cap" : "market_cap",
                "Quotes_BTC_quote_USD_market_cap_dominance" : "market_cap_dominance",
                "Quotes_BTC_quote_USD_percent_change_1h" : "percent_change_1h",
                "Quotes_BTC_quote_USD_percent_change_24h" : "percent_change_24h",
                "Quotes_BTC_quote_USD_percent_change_30d" : "percent_change_30d",
                "Quotes_BTC_quote_USD_percent_change_60d" : "percent_change_60d",
                "Quotes_BTC_quote_USD_percent_change_7d" : "percent_change_7d",
                "Quotes_BTC_quote_USD_percent_change_90d" : "percent_change_90d",
                "Quotes_BTC_quote_USD_price" : "quote_USD_price",
                "Quotes_BTC_quote_USD_volume_24h" : "quote_volume_24h",
                "Quotes_BTC_quote_USD_volume_change_24h" : "volume_change_24h",
                "Quotes_BTC_total_supply" : "total_supply",
                "Ticker_ask" : "ask_24h",  # OHLC ask last 24h
                "Ticker_bid" : "bid_24h",  # OHLC bid last 24h
                "Ticker_high" : "high_24h",  # OHLC high last 24h
                "Ticker_last" : "last_24h",  # OHLC last last 24h
                "Ticker_low" : "low_24h",  # OHLC low last 24h
                "Ticker_open" : "open_24h",  # OHLC open last 24h
                "Ticker_timestamp" : "unix_timestamp",
                "Ticker_volume" : "volume_24h",  # OHLC volume last 24h
                "Ticker_vwap" : "vwap_24h",  # OHLC vwap last 24h
            },
            inplace=True)

        # store flatten df to CSV
        if shouldCreateCSV :
            flatten_df.to_csv(file_name, sep=',', encoding='utf-8', index=False)
    else :
        flatten_df = pd.read_csv(file_name, index_col=0)

    # check for null values per column
    print("NaN values per column count: \n")
    print(flatten_df.isna().sum())

    """
        ### Fix NaN Values.
    
        We observe that Quotes_BTC API returns null some times. 
        We have to fix this missing values.
        First we create a DF with Null values to  
        
        # Columns With NAN values
        
        Quotes_BTC_max_supply                            This value is Constant 21000000.00000
        Quotes_BTC_circulating_supply                    can be filed from previous one
        Quotes_BTC_total_supply                          can be filed from previous one
        Quotes_BTC_quote_USD_price                       #SOS This is difficult malon thelw ena random num mesa sto OHLC High kai low
        Quotes_BTC_quote_USD_volume_24h                  can be filed from previous one
        Quotes_BTC_quote_USD_volume_change_24h           can be filed from previous one
        Quotes_BTC_quote_USD_percent_change_1h           can be filed from previous one
        Quotes_BTC_quote_USD_percent_change_24h          can be filed from previous one
        Quotes_BTC_quote_USD_percent_change_7d           can be filed from previous one
        Quotes_BTC_quote_USD_percent_change_30d          can be filed from previous one
        Quotes_BTC_quote_USD_percent_change_60d          can be filed from previous one
        Quotes_BTC_quote_USD_percent_change_90d          can be filed from previous one
        Quotes_BTC_quote_USD_market_cap                  can be filed from previous one
        Quotes_BTC_quote_USD_market_cap_dominance        can be filed from previous one
        Quotes_BTC_quote_USD_fully_diluted_market_cap    can be filed from previous one
        Quotes_BTC_quote_USD_last_updated                Dont need this column.
    """
    isNullDf = flatten_df[flatten_df.isnull().sum(1) > 0]
    pprint(isNullDf.head())

    # Use Fill Forward to fill the rest of columns
    # flatten_df.fillna(method='ffill', inplace=True)
    flatten_df = flatten_df.interpolate(limit_direction="forward")

    # check for null values per column after interpolate
    print("NaN values per column count: \n")
    print(flatten_df.isna().sum())

    fill_nan_df = flatten_df.iloc[isNullDf.index]
    pprint(fill_nan_df.head())

    # Plot to detect outliers
    # if doPlots :
    #     for col in flatten_df.columns.values.tolist() :
    #         boxplot = flatten_df.boxplot(column=[col])
    #         plt.show()

    # Clean outliers if exist.
    column_list = [
        "close_1min",
        "high_1min",
        "low_1min",
        "open_1min",
        "volume_1min",
        "fully_diluted_market_cap",
        "market_cap",
        "market_cap_dominance",
        "percent_change_1h",
        "percent_change_24h",
        "percent_change_30d",
        "percent_change_60d",
        "percent_change_7d",
        "percent_change_90d",
        "quote_USD_price",
        'quote_volume_24h',
        "volume_change_24h",
        "ask_24h",  # OHLC ask last 24h
        "bid_24h",  # OHLC bid last 24h
        "high_24h",  # OHLC high last 24h
        "last_24h",  # OHLC last last 24h
        "low_24h",  # OHLC low last 24h
        "open_24h",  # OHLC open last 24h
        "volume_24h",  # OHLC volume last 24h
        "vwap_24h",  # OHLC vwap last 24h
    ]
    flatten_df = cleanOutliers(flatten_df, column_list)

    # check for null values per column after interpolate
    print("NaN values per column count after Remove outliers: \n")
    print(flatten_df.isna().sum())

    # check nan leftovers
    isNullDf = flatten_df[flatten_df.isnull().sum(1) > 0]
    pprint(isNullDf.head())

    # Plot to detect outliers
    if doPlots :
        for col in flatten_df.columns.values.tolist() :
            boxplot = flatten_df.boxplot(column=[col])
            plt.show()

    # get reddit posts from mongo
    reddit_df = getRedditPostsFromMongo()
    reddit_df['weighted_polarity'] = reddit_df["compound"] * reddit_df['score']
    print(reddit_df["weighted_polarity"])

    # Get reddit compound polarity for each row
    flatten_df["reddit_compound_polarity"] = flatten_df.apply(lambda row: getWeightedRedditPolarity(
        df=reddit_df,
        timeFrom=row['unix_timestamp'],
        timeRange=30 * ONE_MINUTE_SECONDS
    ), axis=1)

    print(flatten_df["reddit_compound_polarity"])

    # store data to mongo
    storeFlattenData(flatten_df.to_dict('records'))
