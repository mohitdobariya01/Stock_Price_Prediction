import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import pickle
from keras.models import load_model
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

st.set_page_config(page_title='Stock Price Prediction', layout='centered')
st.title('Next Day Stock Price Prediction')

model = load_model('model.h5')

with open('scaler.pkl','rb') as f:
    scaler = pickle.load(f)

ticker = st.text_input("Enter Stock Symbol","RELIANCE.NS")

if st.button("Predict Stock Price"):
    end_date = datetime.today()
    start_date = datetime(end_date.year-10, end_date.month, end_date.day)

    df = yf.download(ticker, start=start_date, end=end_date)

    if df is None or df.empty:
        st.error("❌ No data found. Check stock symbol.")
        st.stop()

    if 'Close' not in df.columns:
        st.error("❌ Close price not available.")
        st.stop()

    if len(df) < 200:
        st.error("❌ Minimum 200 days of data required.")
        st.stop()

    st.subheader("📊Previous Trading Days Data")
    st.dataframe(df.tail(5))

    close_prices = df[['Close']]
    scaled_data = scaler.transform(close_prices)

    lookback = 200
    last_200_days = scaled_data[-lookback:]
    X_test = np.reshape(last_200_days, (1,lookback,1))

    predicted_price = model.predict(X_test)
    predicted_price = scaler.inverse_transform(predicted_price)

    st.subheader("Next Day Predicted Price")
    st.success(f"Rs.{predicted_price[0][0]:.2f}")

    st.subheader("📈 Stock Price Chart")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df['Close'], label="Historical Close Price")
    ax.scatter(df.index[-1] + timedelta(days=1),
            predicted_price[0][0],
            color="red",
            label="Next Day Prediction")

    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid()

    st.pyplot(fig)