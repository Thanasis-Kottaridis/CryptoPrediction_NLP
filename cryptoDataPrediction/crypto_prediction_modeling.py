# Basic Imports
import utils
import random
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# Standardization Imports
from sklearn.preprocessing import MinMaxScaler
# Evaluation Imports
from sklearn.metrics import mean_squared_error
from math import sqrt
# Deep learning import
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential

# Check GPU
import tensorflow as tf
from tensorflow.python.client import device_lib

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(device_lib.list_local_devices())

# Constants
DATA_FILE_NAME = 'final_crypto_data.csv'
readFromMongo = False
isDemoMode = True
doPlots = True
loadModel = False


def toMultivariateSeries(input_sequences, output_sequence, n_steps_in, n_steps_out):
    """
        Split timeSeries
        This helper function is used in order to split dataset in time series.
        split a multivariate sequence past, future samples (X and y)
    """
    X_list, y_list = list(), list()  # instantiate X and y
    for i in range(len(input_sequences)):
        # find the end of the input, output sequence
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out - 1
        # check if we are beyond the dataset
        if out_end_ix > len(input_sequences): break
        # gather input and output of the pattern
        seq_x, seq_y = input_sequences[i:end_ix], output_sequence[end_ix - 1:out_end_ix, -1]
        X_list.append(seq_x), y_list.append(seq_y)
    return np.array(X_list), np.array(y_list)


def train_test_valid_split(total_x, total_y, train_size=0.9, valid_size=0.1):
    train_index = int(len(total_x) * train_size)
    valid_index = int(len(total_x) * valid_size)

    X_train, y_train = total_x[0:train_index], total_y[0:train_index]
    X_valid, y_valid = total_x[train_index:train_index + valid_index], total_y[train_index:train_index + valid_index]
    X_test, y_test = total_x[train_index + valid_index:], total_y[train_index + valid_index:]

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


def getModelCheckpointCallback(checkpoint_path):
    checkpoint_dir = os.path.dirname(checkpoint_path)
    # Create a callback that saves the model's weights
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        verbose=1
    )


def display_training_curves(training, validation, title, subplot=None):
    if not doPlots:
        return
    if subplot is not None:
        ax = plt.subplot(subplot)
    else:
        ax = plt.subplot()
    ax.plot(training)
    ax.plot(validation)
    ax.set_title('model ' + title)
    ax.set_ylabel(title)
    ax.set_xlabel('epoch')
    ax.legend(['training', 'validation'])
    plt.show()
    return


if __name__ == '__main__':
    # Get Processed Crypto Data
    # - Get Data from mongo
    # - store data in DF

    if readFromMongo:
        # connecting or switching to the database
        connection, db = utils.connectMongoDB()

        # creating or switching to ais_navigation collection
        collection = db.flatten_crypto_data

        # Read preprocessed data from mongo
        res = collection.find()

        jsonData = list(res)

        df = pd.DataFrame.from_dict(jsonData)
        # settings to display all columns
        pd.set_option("display.max_columns", None)
        # display the dataframe head
        df.head(5)
    else:
        df = pd.read_csv(DATA_FILE_NAME)

    # Plot Bitcoin Price from collected data

    # Increase size of plot in jupyter
    if doPlots:
        plt.rcParams["figure.figsize"] = (10, 8)
        plt.plot(df['close_1min'])
        plt.xlabel("Observations")
        plt.ylabel("Price (USD)")
        plt.title("Bitcoin price over time")
        plt.savefig("initial_plot.png", dpi=250)
        plt.show()

    # Clean Data
    # - Drop id column
    # - Create X df with all predictors except our target price.
    # - And Y df with target feature ***close_1min***.

    # drop _id column
    df.drop(columns=['_id'], inplace=True)

    # drop unnecessary features
    df.drop(columns=[
        'high_24h',
        'last_24h',
        'unix_timestamp',
        'bid_24h',
        'vwap_24h',
        'volume_24h',
        'low_24h',
        'ask_24h',
        'open_24h',
        # 'high_1min',
        # 'volume_1min',
        # 'low_1min',
        # 'close_1min',
        # 'open_1min',
        'max_supply',
        'circulating_supply',
        'total_supply',
        'quote_USD_price',
        'quote_volume_24h',
        'volume_change_24h',
        'percent_change_1h',
        'percent_change_24h',
        'percent_change_7d',
        'percent_change_30d',
        'percent_change_60d',
        'percent_change_90d',
        'market_cap',
        'market_cap_dominance',
        'fully_diluted_market_cap',
        # 'reddit_compound_polarity'
    ], inplace=True)

    # ensure all data are numeric
    df = df.apply(pd.to_numeric)
    # separate data tou X and y
    X, y = df.drop(columns=['close_1min']), df['close_1min'].values
    print(X.shape, y.shape)

    """
        Standardising Our Features

        Standardisation helps deep learning model training by ensuring
        that parameters can exist in the same multi dimensional space

        - We will use standardisation for our predictor by
          removing mean and scaling to unit variance.
    """

    # normalize features
    scaler = MinMaxScaler()
    X_trans = scaler.fit_transform(X)
    y_trans = scaler.fit_transform(y.reshape(-1, 1))

    """
        # Convert predicators to a list of time series. 
        - Each element of X_series multivariate time series contains 29 predictor features 
          and 120 observations (120 minutes = 2H)
        - Each element o y_series is a univariant time serieas that contains 15 observations 
          of bitcoin price ( target prediction interval 15 min)
    """
    X_series, y_series = toMultivariateSeries(X_trans, y_trans, 30, 1)
    print(X_series.shape, y_series.shape)

    # Split Train Test Dataset
    # To speed up model designing we will use only 20% of dataset
    # in order to see how LSTM Algorithm works on the Data.
    cutoff = round(0.95 * len(X_series)) if isDemoMode else 0
    X_demo_series = X_series[cutoff:]
    y_demo_series = y_series[cutoff:]

    train_X, train_y, valid_X, valid_y, _, _ = train_test_valid_split(
        X_demo_series,
        y_demo_series,
        train_size=0.8,
        valid_size=0.1
    )
    print("----------------------------")
    print(f"-------- Train X: {train_X.shape}, Train y: {train_y.shape} --------")
    print(f"-------- Valid X: {valid_X.shape}, Valid y: {valid_y.shape} --------")
    # print(f"-------- Test X: {test_X.shape}, Test y: {test_y.shape} --------")
    print("----------------------------\n")

    # Network Configurations
    EPOCHS = 250
    BATCH_SIZE = 32
    TRAIN_COUNT = len(train_X)
    VAL_COUNT = len(valid_X)
    val_dataset = (valid_X, valid_y)

    # Design Network
    # TODO MAKE IT A CLASS.
    model = Sequential()
    model.add(LSTM(
        2 * BATCH_SIZE,
        return_sequences=True,
        # activation='relu',
        # unroll=True,
        input_shape=(train_X.shape[1], train_X.shape[2]),
    ))
    model.add(LSTM(
        BATCH_SIZE,
        # activation='relu',
        # unroll=True,
    ))
    model.add(Dense(BATCH_SIZE))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # model = Sequential([
    #     # BatchNormalization(),
    #     LSTM(
    #         units=64,
    #         return_sequences=True,
    #         unroll=True,
    #         activation="relu",
    #         input_shape=(train_X.shape[1], train_X.shape[2])
    #     ),
    #     Dropout(0.2),
    #     LSTM(
    #         units=50,
    #         unroll=True,
    #         activation="relu",
    #         return_sequences=True
    #     ),
    #     Dropout(0.2),
    #     LSTM(
    #         units=50,
    #         unroll=True,
    #         activation="relu"
    #     ),
    #     Dropout(0.2),
    #     Dense(units=1, activation="relu")
    # ])

    # # model.add(Dropout(0.2))
    # # model.add(LSTM(units=50, return_sequences=True))
    # # model.add(Dropout(0.2))
    # # model.add(LSTM(units=50))
    # # model.add(Dropout(0.2))
    # # model.add(Dense(units=256))
    # # model.add(Dense(units=128))
    # model.add(Dropout(0.5))
    # model.add(Dense(units=1, activation="sigmoid"))

    # compile model
    # model.compile(optimizer="adam", loss="mean_squared_error")
    model.compile(loss='mae', optimizer='adam')  # Use Adam for adaptive learning rate during training.
    # opt = keras.optimizers.Adam(learning_rate=0.01)
    # model.compile(optimizer=opt,
    #               loss='mae',
    #               metrics=['mse'])
    model.summary()

    """
        Model directories:
        - training_demo_1min
        - training_demo2_1min
        - training_demo3_1min_multiple_LSTM
        - training_demo_15min
        - training_demo3_15min_multiple_LSTM
        - training_demo_30min
        - training_demo_60min
    """
    checkpoint_dir = "cryptoPrediction/training_demo_1min"
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)

    if latest_checkpoint is not None and loadModel:
        # Load the previously saved weights
        model.load_weights(latest_checkpoint)

        # Re-evaluate the model
        # loss, acc = model.evaluate(valid_X, valid_y, verbose=2)
        #
        # # display training curves
        # display_training_curves([], loss, 'loss')

    else:
        # fit network
        history = model.fit(
            train_X,
            train_y,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(valid_X, valid_y),
            verbose=2,
            shuffle=False,
            callbacks=[getModelCheckpointCallback(checkpoint_dir + "/cp-{epoch:04d}.ckpt")] if loadModel else None
            # add the file name and save epoch to file name
        )

        # display training curves
        display_training_curves(history.history['loss'], history.history['val_loss'], 'loss')

    # make a prediction
    y_predict = model.predict(valid_X)
    # invert scaling for forecast
    inv_y_predict = scaler.inverse_transform(y_predict)
    # invert scaling for actual
    # test_y = test_y.reshape((len(test_y), 1))
    inv_y = scaler.inverse_transform(valid_y)
    # calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y, inv_y_predict))
    print('Test RMSE: %.3f' % rmse)

    # plot
    # max_range = 5
    # for i in range(0, max_range):
    #     value = random.randint(0, len(inv_y_predict))
    #     display_training_curves(inv_y[value], inv_y_predict[value], 'predicted vs actual price')

    display_training_curves(inv_y, inv_y_predict, 'predicted vs actual price')
