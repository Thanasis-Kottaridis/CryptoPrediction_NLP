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

