from email.mime import image
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
import os
from twilio.rest import Client
import twiliokeys as tk

client = Client(tk.accountSid, tk.authToken)

start = '2015-01-01'
end = '2022-04-23'

st.title('Forecast')
st.subheader('Using AI to predict the stock market')
st.text('Created by Grant Hough')

stocks = ("VTI", "AAPL", "GOOG", "FB", "AMD", "ADBE", "TSLA", "INTC", "NFLX", "GOOGL", "MSFT", "PCG", "PSTG", "SOFI", "TWLO", "TWTR")
symbolInput = st.selectbox("Select a stock to analyze", stocks)
df = pd.read_csv(symbolInput + ".csv")


st.subheader('Data from January 1, 2015 to April 23, 2022')
st.write(df.head(10000))

# st.subheader('Closing Prices over Time')
# fig = plt.figure(figsize = (12, 6))
# plt.plot(df.Close)
# st.pyplot(fig)

model = keras.models.load_model('model.h5')

trainingSet = df.iloc[:1000, 1:2].values
testSet = df.iloc[1000:, 1:2].values

sc = MinMaxScaler(feature_range = (0, 1))
trainingSet_scaled = sc.fit_transform(trainingSet)

datasetTrain = df.iloc[:1000, 1:2]
datasetTest = df.iloc[1000:, 1:2]
datasetTotal = pd.concat((datasetTrain, datasetTest), axis = 0)
inputs = datasetTotal[len(datasetTotal) - len(datasetTest) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
XTest = []
for i in range(60, 900):
    XTest.append(inputs[i-60:i, 0])
XTest = np.array(XTest)
XTest = np.reshape(XTest, (XTest.shape[0], XTest.shape[1], 1))

predictedStockPrice = model.predict(XTest)
predictedStockPrice = sc.inverse_transform(predictedStockPrice)

st.subheader('Predicted Prices vs. Actual Prices')
fig = plt.figure(figsize = (12, 6))
plt.plot(df.loc[1000:, 'Date'],datasetTest.values, color = (0.05, 0.32, 0.73), label = 'Actual Stock Price')
plt.plot(df.loc[1000:, 'Date'],predictedStockPrice, color = (0.2, 0.5, 0.6), label = 'Predicted Stock Price')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Stock Price ($)')
plt.xticks(np.arange(0, 1000, 100))
ax = plt.gca()
ax.set_facecolor((0.08, 0.08, 0.1))
fig.patch.set_facecolor((0.13, 0.18, 0.25))
ax.xaxis.label.set_color('white')
ax.tick_params(axis='x', colors='white')
ax.yaxis.label.set_color('white')
ax.tick_params(axis='y', colors='white')
st.pyplot(fig)

startDate = "2015-01-01"
todayDate = date.today().strftime("%Y-%m-%d")
years = st.slider("Number of Future Years to Forcast", 1, 5)
period = years * 365

@st.cache
def loadData(ticker):
    data = yf.download(ticker, startDate, todayDate)
    data.reset_index(inplace = True)
    return data

data = loadData(symbolInput)

DFTrain = data[['Date', 'Close']]
DFTrain = DFTrain.rename(columns = {"Date": "ds", "Close": "y"})

model2 = Prophet()
model2.fit(DFTrain)
future = model2.make_future_dataframe(periods = period)
forecast = model2.predict(future)

st.subheader('Forecasted Prices')
forecastFig = model2.plot(forecast)
ax = forecastFig.gca() 
ax.set_xlabel("Time", size=10)
ax.set_ylabel("Stock Price ($)", size=10)
ax.set_facecolor((0.08, 0.08, 0.1))
forecastFig.patch.set_facecolor((0.13, 0.18, 0.25))
ax.xaxis.label.set_color('white')
ax.tick_params(axis='x', colors='white')
ax.yaxis.label.set_color('white')
ax.tick_params(axis='y', colors='white')
st.pyplot(forecastFig)

st.caption('The black dots indicate real closing prices, and the blue line indicates the prediction of the model. The area without black dots is the prediction of closing prices that the model created.')

st.subheader('Text the Creator Feedback')

textInput = ''
textMessage = st.text_input("Enter a message", textInput)

def sendMessage():
    if (len(textMessage) > 0):
        message = client.messages.create (
            body = textMessage,
            from_ = tk.twilioNumber,
            to = tk.targetNumber
        )
  
st.button("Send Message", on_click = sendMessage)