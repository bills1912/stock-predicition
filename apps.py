import warnings
import numpy as np
import pandas as pd
import datetime as dt
import streamlit as st
import tensorflow as tf
import pandas_datareader as pdr
from tensorflow.keras import load_model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

# warnings.filterwarnings('ignore')
# plt.style.use('fivethirtyeight')

st.set_page_config(
        page_title="FANG Stock Prediction",
        page_icon=":bitcoin:",
        layout="wide"
    )

st.write("# Welcome to FANG Stock Prediction Dashboard :bitcoin:")

company_ticker = ['FB', 'AAPL', 'TSLA', 'GOOG', 'NVDA']

start_date = dt.datetime(2007, 1, 1)
end_date = dt.datetime(2020, 12, 13)
data_FB = pdr.DataReader(company_ticker[0], 'yahoo', start_date, end_date)
# data_AAPL = pdr.DataReader(company_ticker[1], 'yahoo', start_date, end_date)
# data_TSLA = pdr.DataReader(company_ticker[2], 'yahoo', start_date, end_date)
data_GOOG = pdr.DataReader(company_ticker[3], 'yahoo', start_date, end_date)
# data_NVDA = pdr.DataReader(company_ticker[4], 'yahoo', start_date, end_date)

scaler = MinMaxScaler()
scaled_data_FB = scaler.fit_transform(data_FB.filter(['Adj Close']).values.reshape(-1, 1))
scaled_data_GOOG = scaler.fit_transform(data_GOOG.filter(['Adj Close']).values.reshape(-1, 1))
prediction_days = 89

test_start = dt.datetime(2020, 12, 31)
test_end = dt.datetime.now()
test_data_FB = pdr.DataReader(company_ticker[0], 'yahoo', test_start, test_end)
# test_data_AAPL = pdr.DataReader(company_ticker[1], 'yahoo', test_start, test_end)
# test_data_TSLA = pdr.DataReader(company_ticker[2], 'yahoo', test_start, test_end)
test_data_GOOG = pdr.DataReader(company_ticker[3], 'yahoo', test_start, test_end)
# test_data_NVDA = pdr.DataReader(company_ticker[4], 'yahoo', test_start, test_end)
actual_prices = test_data_GOOG.filter(['Adj Close']).values

total_dataset = pd.concat((data_GOOG['Adj Close'], test_data_GOOG['Adj Close']), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data_GOOG) - prediction_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.fit_transform(model_inputs)

def test_data_predict(lstm_model):
  X_test = []

  for x in range(prediction_days, len(model_inputs)):
    X_test.append(model_inputs[x-prediction_days:x, 0])
  
  X_test = np.array(X_test)
  X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

  prediction_prices = lstm_model.predict(X_test)
  prediction_prices = scaler.inverse_transform(prediction_prices)

  return prediction_prices

stock_lstm = load_model("fang_stock_prediction.h5")

predict_prices = test_data_predict(lstm_model=stock_lstm)
valid = test_data_GOOG.filter(['Adj Close'])
valid['Predictions'] = predict_prices

st.line_chart(data_GOOG.filter['Adj Close'])
st.line_chart(valid)

real_data = [model_inputs[len(model_inputs)+1 - prediction_days:len(model_inputs)+1, 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

real_prediction = stock_lstm.predict(real_data)
real_prediction = scaler.inverse_transform(real_prediction)
st.text(f"Real prediction stock prices on Google is {real_prediction[0]}")