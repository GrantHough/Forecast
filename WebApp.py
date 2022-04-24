from tkinter import TRUE
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pandas_datareader as data
import keras
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
import yfinance as yf
from datetime import date
import plotly.offline as py
import plotly.graph_objs as go
import ipywidgets

start = '2015-01-01'
end = '2022-04-23'

st.title('Stock Trend Prediction')
st.text('Created by Grant Hough')

stocks = ("VTI", "AAPL", "GOOG", "FB", "AMD", "ADBE", "TSLA", "INTC")
symbolInput = st.selectbox("Select a stock to analyze", stocks)
df = pd.read_csv(symbolInput + ".csv")


st.subheader('Data from January 1, 2015 to April 23, 2022')
st.write(df.head(10000))

# st.subheader('Closing Prices over Time')
# fig = plt.figure(figsize = (12, 6))
# plt.plot(df.Close)
# st.pyplot(fig)

model = keras.models.load_model('model.h5')

training_set = df.iloc[:1000, 1:2].values
test_set = df.iloc[1000:, 1:2].values

sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

dataset_train = df.iloc[:1000, 1:2]
dataset_test = df.iloc[1000:, 1:2]
dataset_total = pd.concat((dataset_train, dataset_test), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 900):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

st.subheader('Predicted Prices vs. Actual Prices')
fig = plt.figure(figsize = (12, 6))
plt.plot(df.loc[1000:, 'Date'],dataset_test.values, color = 'cyan', label = 'Actual Stock Price')
plt.plot(df.loc[1000:, 'Date'],predicted_stock_price, color = 'blue', label = 'Predicted Stock Price')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Stock Price ($)')
plt.xticks(np.arange(0, 1000, 100))
st.pyplot(fig)



startDate = "2015-01-01"
todayDate = date.today().strftime("%Y-%m-%d")
years = st.slider("Years of Prediction", 1, 5)
period = years * 365

@st.cache
def loadData(ticker):
    data = yf.download(ticker, startDate, todayDate)
    data.reset_index(inplace = True)
    return data

data = loadData(symbolInput)

df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns = {"Date": "ds", "Close": "y"})

model2 = Prophet()
model2.fit(df_train)
future = model2.make_future_dataframe(periods = period)
forecast = model2.predict(future)

st.subheader('Forecasted Prices')
forecastFig = model2.plot(forecast)
ax = forecastFig.gca() 
ax.set_xlabel("Time", size=10)
ax.set_ylabel("Stock Price ($)", size=10)
st.pyplot(forecastFig)

