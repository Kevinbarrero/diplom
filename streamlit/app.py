import streamlit as st
from keras.models import load_model
from datetime import datetime, timedelta
import yfinance as yf
from features import *
import joblib
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


st.markdown(
    """
    <style>
    .normal-font {
        text-align: center; 
        font-size: 24px;
        font-weight: normal;
    }
    </style>
    """, 
    unsafe_allow_html=True
)

st.markdown("<h1 class='normal-font'>Crypto Mind</h1>", unsafe_allow_html=True)

today = datetime.now().date()
next_day = today + timedelta(days=1)

options = [
    'S&P 500 Index', 'Bitcoin (USD)', 'Ethereum (USD)', 'Apple Inc.', 'Microsoft Corporation', 
    'Amazon.com, Inc.', 'Alphabet Inc. (Google)', 'Tesla, Inc.', 'Meta Platforms, Inc. (Facebook)', 
    'Netflix, Inc.', 'Johnson & Johnson', 'Visa Inc.', 'JPMorgan Chase & Co.', 'Walmart Inc.', 
    'Procter & Gamble Co.', 'Exxon Mobil Corporation', 'SPDR Gold Trust (Gold ETF)', 
    'iShares Silver Trust (Silver ETF)', 'United States Oil Fund, LP (Oil ETF)', 
    'Dow Jones Industrial Average', 'NASDAQ Composite Index', 'Russell 2000 Index', 
    'SPDR S&P 500 ETF', 'iShares S&P 500 ETF', 'Vanguard S&P 500 ETF'
]

assets = {
    'S&P 500 Index': '^GSPC', 'Bitcoin (USD)': 'BTC-USD', 'Ethereum (USD)': 'ETH-USD', 'Apple Inc.': 'AAPL', 
    'Microsoft Corporation': 'MSFT', 'Amazon.com, Inc.': 'AMZN', 'Alphabet Inc. (Google)': 'GOOGL', 
    'Tesla, Inc.': 'TSLA', 'Meta Platforms, Inc. (Facebook)': 'META', 'Netflix, Inc.': 'NFLX', 
    'Johnson & Johnson': 'JNJ', 'Visa Inc.': 'V', 'JPMorgan Chase & Co.': 'JPM', 'Walmart Inc.': 'WMT', 
    'Procter & Gamble Co.': 'PG', 'Exxon Mobil Corporation': 'XOM', 'SPDR Gold Trust (Gold ETF)': 'GLD', 
    'iShares Silver Trust (Silver ETF)': 'SLV', 'United States Oil Fund, LP (Oil ETF)': 'USO', 
    'Dow Jones Industrial Average': '^DJI', 'NASDAQ Composite Index': '^IXIC', 'Russell 2000 Index': '^RUT', 
    'SPDR S&P 500 ETF': 'SPY', 'iShares S&P 500 ETF': 'IVV', 'Vanguard S&P 500 ETF': 'VOO'
}


default_option = options[1]  # Choose the first option as default
selected_option = st.selectbox('пожалуйста, выберите актив, чтобы получить прогноз', options, index=options.index(default_option))


stock_symbol = assets[selected_option]

stock = yf.Ticker(stock_symbol)
df = stock.history(period="1y")
df = df.drop(columns=['Dividends', 'Stock Splits'])

df['open'] = df['Open']
df['high'] = df['High']
df['low'] = df['Low']
df['close'] = df['Close']
df['volume'] = df['Volume']

stats = generate_stats(df)
patterns = generate_patterns(df)
cycles = generate_cycles(df)
indicators = generate_ind(df)

df = pd.concat([df, patterns], axis=1)
df = pd.concat([df, cycles], axis=1)
df = pd.concat([df, stats], axis=1)
df = pd.concat([df, indicators], axis=1)
df = df[100:]

sc = joblib.load('scaler2.pkl')
columns_norm = cycles.columns.to_list() + indicators.columns.to_list() 
df[columns_norm] = sc.transform(df[columns_norm])

features = ['high','low','close'] +  stats.columns.to_list()[:-1] + columns_norm + patterns.columns.to_list()
X_lin = df[features].values.reshape(df[features].shape[0], 1, df[features].shape[1])
Y_lin = df['next_day_close'].values

model = load_model('best_price4.keras', custom_objects={'r2_metric': r2_metric})
if stock_symbol != '^GSPC' or stock_symbol != 'BTC-USD' or stock_symbol != 'ETH-USD':
    model = load_model('best_price5.keras', custom_objects={'r2_metric': r2_metric})

df['predictions'] = model.predict(X_lin)
df = df.tail(90)

predictions = df['predictions']
Y_test = df['next_day_close']


plt.figure(figsize=(10, 5))
plt.plot(df.index, df.next_day_close, label='фактическая цена')
plt.plot(df.index[:-1], predictions[:-1], color='red', label='прогноз')


last_prediction = predictions.iloc[-1]
error_margin = 0.02 * last_prediction  # Example: 2% error margin
plt.errorbar(df.index[-1], last_prediction, yerr=error_margin, fmt='o', color='red', ecolor='green', label='Диапазон следующего дня')

plt.xlabel('Дата')
plt.ylabel('Цена')
plt.xticks(rotation=45)
plt.legend()


st.markdown("<h1 class='normal-font'>Прогноз на завтра</h1>", unsafe_allow_html=True)
tomorrow = df['predictions'].iloc[-1]
st.markdown(f"<h1 class='normal-font'>{tomorrow:.2f}</h1>", unsafe_allow_html=True)


def add_percentage(value, percentage):
    return value * (1 + percentage / 100)


st.write('Минимальная цена:', add_percentage(tomorrow, -2))
st.write('Максимальная цена:', add_percentage(tomorrow, 2))


current_price = df['close'].iloc[-1]
recent_high = df['close'].max()
recent_low = df['close'].min()
st.write('Текущая цена:', current_price)
st.write('Недавний максимум:', recent_high)
st.write('Недавний минимум:', recent_low)

df['returns'] = df['close'].pct_change()

mu, std = norm.fit(df['returns'].dropna())


predicted_change = (tomorrow - current_price) / current_price

z_score = (predicted_change - mu) / std

probability_rise = norm.cdf(z_score)
probability_drop = 1 - probability_rise


st.markdown("<h1 class='normal-font'>прогнозы на последние три месяца</h1>", unsafe_allow_html=True)
st.pyplot(plt)

gauge_chart = plot_gauge(probability_rise*100, 'вероятность роста')
st.plotly_chart(gauge_chart)
    
recent_days = 7
recent_trend = df['close'].tail(recent_days).pct_change().mean() * 100
st.write(f'Среднее изменение цены за последние {recent_days} дней: {recent_trend:.2f}%')

volatility = df['close'].pct_change().std() * 100
st.write(f'Волатильность за последний месяц: {volatility:.2f}%')


