import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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


"""
    Dataframe Preparation Utils
"""


def setDateTimeAsIndex(df, unixTimestampColumn='unix_timestamp', targetName='datetime'):
    """
    Use this function to convert unix timestamp to datetime and set it as index in targer DF
    :param df: target Dataframe
    :param unixTimestampColumn: initial unix timestamp column name
    :param targetName: target name for datetime column.
    :return: dataframe after changes applied.
    """
    df[unixTimestampColumn] = pd.to_datetime(df[unixTimestampColumn], unit='s')
    df.rename(columns={unixTimestampColumn : targetName}, inplace=True)
    df.set_index(targetName, inplace=True)
    return df


def filterColumns(df, selectedColumns, reindex=True):
    """
    Use this function to keep selected columns and change their order
    :param df: target pandas dataframe
    :param selectedColumns: a list that contains target column names
    :param reindex: boolean flag that indicated if reindex will apply default [True]
    :return: returns dataframe after transformations
    """
    df = df.drop(columns=[col for col in df if col not in selectedColumns])
    if reindex:
        df = df.reindex(columns=selectedColumns)
    return df


def filterInMonths(df, selectedMonths):
    """
    This helper func is used to filter df rows in month,
    @SOS: this function assumes that date time is the index of the dataframe
    :param df: target pandas dataframe
    :param selectedMonths: a  list that contains target months in integers 1 -> January 2 -> february etc.
    :return: returns dataframe after transformations
    """
    targetIndexes = [i for i in df.index if i.month in selectedMonths]
    filter_df = df[df.index.isin(targetIndexes)]
    return filter_df


def train_test_valid_split(total_x, total_y, train_size=0.9, valid_size=0.1) :
    train_index = int(len(total_x) * train_size)
    valid_index = int(len(total_x) * valid_size)

    X_train, y_train = total_x[0 :train_index], total_y[0 :train_index]
    X_valid, y_valid = total_x[train_index :train_index + valid_index], total_y[train_index :train_index + valid_index]
    X_test, y_test = total_x[train_index + valid_index :], total_y[train_index + valid_index :]

    print("-------- train test valid split --------")
    print(len(X_train)), print(len(y_train))
    print(len(X_valid)), print(len(y_valid))
    print(len(X_test)), print(len(y_test))
    print("----------------------------------------")

    return np.array(X_train), \
           np.array(y_train), \
           np.array(X_valid), \
           np.array(y_valid), \
           np.array(X_test), \
           np.array(y_test)


def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


"""
    Display Data Utils
"""


def display_training_curves(
        training,
        validation,
        title,
        x_title="Minute Observations",
        y_title="Closing Price",
        subplot=None
) :
    plt.figure(figsize=(15, 9))
    plt.grid(True)

    if subplot is not None:
        ax = plt.subplot(subplot)
    else:
        ax = plt.subplot()

    ax.plot(training)
    ax.plot(validation)
    ax.set_title(title)
    plt.xlabel(x_title)
    plt.ylabel(y_title)

    ax.set_ylabel(title)
    ax.set_xlabel('epoch')
    ax.legend(['training', 'validation'])
    plt.show()