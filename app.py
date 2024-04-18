import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import pandas_datareader as data
import yfinance as yf
from keras.models import load_model
import streamlit as st

start = "2010-01-01"
end = "2019-12-31"
    
st.title('Stock Price Prediction')
user_input = st.text_input("Enter Stock Ticker", "AAPL")

df = yf.download(user_input, start, end)

#Describing Data
st.subheader(f'Data from {start}-{end}')
st.write(df.describe())

#Visualization
st.subheader(f'Closing price vs Time chart ({user_input})')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing price vs Time chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing price vs Time chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close, 'b')
st.pyplot(fig)


#Splitting data into train and test
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

#MinMaxScaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)

# Load my model
model = load_model('keras_model.h5')

# Testing part
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
  x_test.append(input_data[i-100: i])
  y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_prediction = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_prediction = y_prediction * scale_factor
y_test = y_test * scale_factor


# Final Graph
st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_prediction, 'r', label='Predicted Price')
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
st.pyplot(fig2)

# Fetch the original closing price for the last day
original_last_day = df['Close'].iloc[-1]

# Display the original closing price for the last day
# st.subheader(f"Original closing price for the last day: {original_last_day}")

# Fetch the predicted value for the last day
predicted_last_day = y_prediction[-1]

# Display the predicted value
st.subheader(f"Predicted closing price for the next day: {predicted_last_day[0].round()}")
