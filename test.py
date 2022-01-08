import utils
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt

_, flatten_data = utils.cleanMongoData()

# create DF with flatten data.
flatten_df = pd.DataFrame.from_dict(flatten_data)
# flatten_df['Quotes_BTC_quote_USD_price'] = flatten_df['Quotes_BTC_quote_USD_price'].round(decimals=2)

# q_low = flatten_df["Quotes_BTC_quote_USD_price"].quantile(0.01)
# q_hi = flatten_df["Quotes_BTC_quote_USD_price"].quantile(0.99)

# df_filtered = flatten_df[(flatten_df["Quotes_BTC_quote_USD_price"] < q_hi) & (flatten_df["Quotes_BTC_quote_USD_price"] > q_low)]

print(flatten_df['Quotes_BTC_quote_USD_price'].max())
print(flatten_df['Quotes_BTC_quote_USD_price'].min())

# pprint(flatten_df.head())
#
# flatten_df['date'] = pd.to_datetime(flatten_df['Ticker_timestamp_iso'], format='%Y-%m-%d %H:%M:%S')
# grouped_df = flatten_df.groupby([flatten_df['date'].dt.date])['Quotes_BTC_quote_USD_price'].mean()
# pprint(grouped_df.head())
# grouped_df.plot.line()
# plt.show()
