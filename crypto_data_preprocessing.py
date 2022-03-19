import utils
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt
import json
from bson.json_util import loads, dumps

# # clean data from mongo
# # and keep only features needed for prediction
# _, flatten_data = utils.cleanMongoData()
#
# # store json file
# utils.mongoDictToJson(flatten_data, "crypto_data.json")
#
# flatten_df = pd.DataFrame.from_dict(flatten_data)

flatten_df = pd.read_json("crypto_data.json")

# check for null values per column
print("NaN values per column count: \n")
print(flatten_df.isna().sum())

"""
    We observe that Quotes_BTC API returns null some times. 
    We have to fix this missing values.
    First we create a DF with Null values   
"""
isNullDf = flatten_df[flatten_df.isnull().sum(1) > 0]
pprint(isNullDf.head())
