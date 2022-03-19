from time import mktime
from pytz import utc, timezone
from datetime import datetime
import json
from bson.json_util import loads, dumps

"""
    Datetime utils
"""


def utc_to_datetime(ts) :
    # if you encounter a "year is out of range" error the timestamp
    # may be in milliseconds, try `ts /= 1000` in that case
    return datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')


def date_to_utc(year, month, day, hour, minute, second) :
    input_date = datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=second)
    print(input_date)
    return mktime(utc.localize(input_date).utctimetuple())


"""
    jSON Utils
"""


def flatten_json(y, targetFields, flatList=False):
    out = {}

    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '_')
        elif type(x) is list and name[:-1] in targetFields:
            if flatList:
                i = 0
                for a in x:
                    flatten(a, name + str(i) + '_')
                    i += 1
            else:
                out[name[:-1]] = x
        elif name[:-1] in targetFields:
            out[name[:-1]] = x

    flatten(y)
    return out


def mongoDictToJson(d, jsonFileName):
    """
    This helper func extracts a python dictionary loaded from mongo (with ObjectID)
    and stores it into a json file.
    :param d: the dictionary.
    :param jsonFileName: the name of the json file included .json extension.
    :return: Void
    """
    # unwrap obj ID and keep str value
    for obj in d:
        obj["_id"] = str(obj["_id"])

    f = open(jsonFileName, "w")
    json.dump(d, f)
    f.close()


"""
    Mongo Utils
"""
from pymongo import MongoClient

# configuration const!
isLocal = True  # False to query server
connect_to_server = 1  # 1 to connect to server .74, 2 to connect to server private network
showQueryExplain = False


def connectMongoDB() :
    try :
        if isLocal :
            # connect to local mongo db
            connect = MongoClient()

            # connecting or switching to the database
            db = connect.crypto_data_warehouse
        else :
            # conect to mongo server
            if connect_to_server == 1 :
                connect = MongoClient("mongodb://mongoadmin2:mongoadmin@83.212.117.74/admin")
            else :
                connect = MongoClient("mongodb://mongoadmin2:mongoadmin@192.168.0.1/admin")

            # connecting or switching to the database
            db = connect.crypto_data_warehouse

        return connect, db
    except :
        print("Could not connect MongoDB")


def cleanMongoData():
    """
    This helper fun is used to clean all crypto data selected from bitcoin API
    and stored into processed_crypto_data collection.

    This functions selects only the most valuable fields and inserts them in to a new
    mongo Collection named flatten_crypto_data

    :return: It returns flatten_data dictionary
    and a separate list with ids (ids, flatten_data)
    """
    target_elements = [
        "OHLC_close",
        "OHLC_high",
        "OHLC_low",
        "OHLC_open",
        "OHLC_timestamp",
        "OHLC_timestamp_iso",
        "OHLC_volume",
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
        "Quotes_BTC_quote_USD_last_updated",
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
        "Ticker_ask",
        "Ticker_bid",
        "Ticker_high",
        "Ticker_last",
        "Ticker_low",
        "Ticker_open",
        "Ticker_timestamp", # keep one timestamp
        "Ticker_timestamp_iso",
        "Ticker_volume",
        "Ticker_vwap",
    ]

    # connecting or switching to the database
    connection, db = connectMongoDB()

    # creating or switching to ais_navigation collection
    collection = db.processed_crypto_data

    # Mongo response
    res = collection.find()

    jsonData = list(res)

    # create flatten data.
    flatten_data = []
    print(len(jsonData))
    for doc in jsonData:
        flatten_data.append(flatten_json(doc, target_elements))

    clean_collection = db.flatten_crypto_data
    # drop collection if exists
    clean_collection.drop()
    # insert new documents
    ids = clean_collection.insert_many(flatten_data)
    return ids, flatten_data
