#==========================================
# Title:  Stock Forecasting model using LSTM networks
# Author: Vachan
# Email : vachan@iitb.ac.in
#==========================================
import yfinance as yf
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn import preprocessing

# Fetch data using yfinance
ticker = input("Enter ticker:")
start_date = "1986-03-13"
final_date ="2024-12-23"
begin_date = input("Enter the start date(yyyy-mm-dd):") #starting date for the model
end_date= input("Enter the end date(yyyy-mm-dd):")
# Download historical data
df = yf.download(ticker, start=start_date, end=final_date)


# Select only 'Close' prices and reset index
df = df[['Close']].reset_index()
df['Date'] = pd.to_datetime(df['Date'])  # Ensure 'Date' is in datetime format
# Retain both 'Date' column and set it as the index
df.set_index('Date', inplace=True)
#df=df.pct_change().dropna() # use this for percentage returns
print(df)

# Function to convert string to datetime for easier manipulation
def str_to_datetime(s):
    split = s.split('-')
    year, month, day = int(split[0]), int(split[1]), int(split[2])
    return datetime.datetime(year=year, month=month, day=day)


# Plot the closing prices
plt.plot(df.index, df['Close'])
plt.title(f"{ticker} Closing Prices")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.show()
# Function to create windowed DataFrame
def df_to_windowed_df(dataframe, first_date_str, last_date_str, n=3):
    first_date = str_to_datetime(first_date_str)
    last_date = str_to_datetime(last_date_str)
    target_date = first_date

    dates = []
    X, Y = [], []

    last_time = False
    while True:
        print(target_date)
        df_subset = dataframe.loc[:target_date].tail(n + 1)
        if len(df_subset) != n + 1:
            print(f'Error: Window of size {n} is too large for date {target_date}')
            return

        values = df_subset['Close'].to_numpy()
        x, y = values[:-1], values[-1]

        dates.append(target_date)
        X.append(x)
        Y.append(y)

        next_week = dataframe.loc[target_date:target_date + datetime.timedelta(days=7)]
        next_datetime_str = str(next_week.head(2).tail(1).index.values[0])
        next_date_str = next_datetime_str.split('T')[0]
        year_month_day = next_date_str.split('-')
        year, month, day = year_month_day
        next_date = datetime.datetime(day=int(day), month=int(month), year=int(year))

        if last_time:
            break

        target_date = next_date

        if target_date == last_date:
            last_time = True

    ret_df = pd.DataFrame({})
    ret_df['Target Date'] = dates

    X = np.array(X)
    for i in range(0, n):
        X[:, i]
        ret_df[f'Target-{n - i}'] = X[:, i]

    ret_df['Target'] = Y

    return ret_df


# Generate the windowed dataset
windowed_df = df_to_windowed_df(df, begin_date, end_date, n=3)

# Convert the windowed DataFrame to arrays for model training
def windowed_df_to_date_X_y(windowed_dataframe):
    df_as_np = windowed_dataframe.to_numpy()
    dates = df_as_np[:, 0]                                             #Selects the first column (Target Date) from the array.
    middle_matrix = df_as_np[:, 1:-1]                                  #Selects columns from index 1 to the second-to-last column (i.e., the feature columns: Target-3, Target-2, Target-1).
    X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1)) #Reshapes the middle_matrix to fit the input format required by recurrent neural networks (RNNs) like LSTMs, 1 is a a third dimension to represent a single "feature channel"
    Y = df_as_np[:, -1]                                                #Selects the last column (Target) from the array
    return dates, X.astype(np.float32), Y.astype(np.float32)

dates, X, y = windowed_df_to_date_X_y(windowed_df)

# Split into training, validation, and test sets
q_80 = int(len(dates) * 0.8)
q_90 = int(len(dates) * 0.9)
df_train = df[:q_80]  # Assuming q_80 is the split index for training
dates_train, X_train, y_train = dates[:q_80], X[:q_80], y[:q_80]
dates_val, X_val, y_val = dates[q_80:q_90], X[q_80:q_90], y[q_80:q_90]
dates_test, X_test, y_test = dates[q_90:], X[q_90:], y[q_90:]

# Build and train the LSTM model
model = Sequential([
    layers.Input((3, 1)),                                                  # 3 timesteps (window size), 1 feature per timestep.
    layers.LSTM(64),                                                       # An LSTM layer with 64 hidden units
    layers.Dense(32, activation='relu'),                                   # ReLU activation introduces non-linearity, enabling the network to model complex patterns
    layers.Dense(32, activation='relu'),
    layers.Dense(1)                                                        # Outputs a single value (the predicted target).
])
# Define EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',   # Monitor validation loss
    patience=10,          # Stop if no improvement for 10 epochs
    restore_best_weights=True  # Revert to the best model weights
)
model.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics=['mean_absolute_error'])
print("Training begins...")
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, callbacks=[early_stopping])

# Make predictions and visualize
train_predictions = model.predict(X_train).flatten()                       # Converts predictions from a 2D array (e.g., [ [value1], [value2] ]) into a 1D array (e.g., [value1, value2]).
val_predictions = model.predict(X_val).flatten()
test_predictions = model.predict(X_test).flatten()

plt.plot(dates_train, train_predictions)
plt.plot(dates_train, y_train)
plt.plot(dates_val, val_predictions)
plt.plot(dates_val, y_val)
plt.plot(dates_test, test_predictions)
plt.plot(dates_test, y_test)
plt.legend(['Training Predictions', 'Training Observations',
            'Validation Predictions', 'Validation Observations',
            'Testing Predictions', 'Testing Observations'])
plt.show()
