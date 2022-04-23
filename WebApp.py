import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pandas_datareader as data
from keras.models import load_model
import streamlit as st

start = '2015-07-09'
end = '2020-07-09'

df = data.DataReader('AAPL', 'TSLA', start, end)
df.head()

