import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import Label, Text, Button
import requests
import threading
import logging
import smtplib
from email.mime.text import MIMEText

# Logging setup
logging.basicConfig(filename='XAUUSD_bot.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# NewsAPI configuration
NEWS_API_KEY = 'ciJ11xRiUciWcF0V7qrV1apg8jk3hlkp2Yd44Kz4'  # Replace with your NewsAPI key
NEWS_API_URL = 'https://api.marketaux.com/v1/news/all'

# MetaTrader5 path
MT5PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"

# Email configuration
EMAIL_SENDER = "beatsbyzare@gmail.com"  # Replace with your email
EMAIL_PASSWORD = "ymbk iskr gykb akor "  # Replace with your email password
EMAIL_RECEIVERS = ["etunuka14@gmail.com", "wilocholla@gmail.com"]  # List of email receivers


# Connect to MetaTrader5
def connect_mt5():
    try:
        if not mt5.initialize(MT5PATH):
            logging.error("MT5 Connection Failed. Check if MT5 is running and the path is correct.")
            return False
        logging.info("✅ Connected to MT5 successfully.")
        return True
    except Exception as e:
        logging.error(f"Error connecting to MT5: {e}")
        return False

# Fetch historical data
def get_xauusd_data():
    try:
        if not connect_mt5():
            return pd.DataFrame()

        # Check available symbols
        symbols = mt5.symbols_get()
        logging.info(f"Available symbols: {[s.name for s in symbols]}")

        # Check if XAUUSD is available
        if "XAUUSDm" not in [s.name for s in symbols]:
            logging.error("XAUUSD not found in available symbols.")
            return pd.DataFrame()

        # Fetch data
        rates = mt5.copy_rates_from_pos("XAUUSDm", mt5.TIMEFRAME_M5, 0, 200)  # Last 200 candles
        df = pd.DataFrame(rates)

        if df.empty:
            logging.error("No data retrieved from MT5.")
            return pd.DataFrame()

        df['time'] = pd.to_datetime(df['time'], unit='s')

        # Validate columns
        if 'close' not in df.columns:
            logging.error(f"Unexpected DataFrame structure: {df.columns}")
            return pd.DataFrame()

        logging.info(f"Fetched data successfully. Last row:\n{df.iloc[-1]}")
        return df
    except Exception as e:
        logging.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# RSI Calculation
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = pd.Series(gain).rolling(window=period, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=1).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# ATR Calculation
def compute_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=1).mean()

# MACD Calculation
def compute_macd(df, fast_period=12, slow_period=26, signal_period=9):
    df['EMA_Fast'] = df['close'].ewm(span=fast_period, adjust=False).mean()
    df['EMA_Slow'] = df['close'].ewm(span=slow_period, adjust=False).mean()
    df['MACD'] = df['EMA_Fast'] - df['EMA_Slow']
    df['MACD_Signal'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()
    return df

# RSI Divergence Check
def check_rsi_divergence(df, period=14):
    rsi = compute_rsi(df['close'], period)
    price_highs = df['high'].rolling(window=period).max()
    price_lows = df['low'].rolling(window=period).min()
    rsi_highs = rsi.rolling(window=period).max()
    rsi_lows = rsi.rolling(window=period).min()

    # Bullish Divergence: Lower lows in price but higher lows in RSI
    if (price_lows.iloc[-1] < price_lows.iloc[-2] and rsi_lows.iloc[-1] > rsi_lows.iloc[-2]):
        return "BULLISH"

    # Bearish Divergence: Higher highs in price but lower highs in RSI
    if (price_highs.iloc[-1] > price_highs.iloc[-2] and rsi_highs.iloc[-1] < rsi_highs.iloc[-2]):
        return "BEARISH"

    return None

# Support/Resistance Levels
def compute_support_resistance(df, period=20):
    df['Support'] = df['low'].rolling(window=period).min()
    df['Resistance'] = df['high'].rolling(window=period).max()
    return df

# Volume Analysis
def compute_volume_ma(df, period=20):
    df['Volume_MA'] = df['tick_volume'].rolling(window=period).mean()
    return df

# Check if market is range-bound
def is_range_bound(df, atr_threshold=1.0):
    atr = compute_atr(df)
    return atr.iloc[-1] < atr_threshold

# Generate Buy/Sell signals (Modified 3/5 conditions version)
def generate_signal(df):
    """
    Generate a trading signal when 3 out of 5 key conditions are met
    """
    # Initialize counters
    buy_conditions_met = 0
    sell_conditions_met = 0

    # Check if the market is range-bound
    if is_range_bound(df):
        return "WAIT", None, None, buy_conditions_met, sell_conditions_met

    # Compute indicators
    df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['close'].ewm(span=200, adjust=False).mean()
    df['RSI'] = compute_rsi(df['close'], 14)
    df['ATR'] = compute_atr(df, 14)
    df = compute_macd(df)
    df = compute_support_resistance(df)
    df = compute_volume_ma(df)

    last_close = df['close'].iloc[-1]
    atr = df['ATR'].iloc[-1]
    rsi_divergence = check_rsi_divergence(df)

    # Key conditions (5 total)
    conditions = {
        'ema_cross': df['EMA_50'].iloc[-1] > df['EMA_200'].iloc[-1],
        'rsi_oversold': df['RSI'].iloc[-1] < 30,
        'macd_cross': df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1],
        'rsi_divergence': rsi_divergence == "BULLISH",
        'volume_spike': df['tick_volume'].iloc[-1] > df['Volume_MA'].iloc[-1]
    }

    # Count buy conditions met
    buy_conditions_met = sum([
        conditions['ema_cross'],
        conditions['rsi_oversold'],
        conditions['macd_cross'],
        conditions['rsi_divergence'],
        conditions['volume_spike']
    ])

    # Count sell conditions met (inverse logic)
    sell_conditions_met = sum([
        not conditions['ema_cross'],
        df['RSI'].iloc[-1] > 70,  # Overbought instead of oversold
        not conditions['macd_cross'],
        rsi_divergence == "BEARISH",
        conditions['volume_spike']
    ])

    # Generate Buy Signal (3/5 conditions)
    if buy_conditions_met >= 3:
        sl = df['Support'].iloc[-1] - 0.5 * atr
        tp = df['Resistance'].iloc[-1] + 0.5 * atr
        return "BUY", sl, tp, buy_conditions_met, sell_conditions_met

    # Generate Sell Signal (3/5 conditions)
    if sell_conditions_met >= 3:
        sl = df['Resistance'].iloc[-1] + 0.5 * atr
        tp = df['Support'].iloc[-1] - 0.5 * atr
        return "SELL", sl, tp, buy_conditions_met, sell_conditions_met

    return "WAIT", None, None, buy_conditions_met, sell_conditions_met

# Fetch Sentiment Data
def fetch_sentiment():
    try:
        params = {
            'q': 'Inflation,Interest rates,Federal Reserve,Central banks,Monetary policy,Quantitative easing,Economic recession,GDP growth,Stock market volatility,Currency fluctuations,U.S. dollar strength,Forex market,Bond yields,Treasury rates,Consumer Price Index (CPI),Producer Price Index (PPI),Employment data,Unemployment rate,reserves',
            'api_token': NEWS_API_KEY,
            'language': 'en',
            'filter_entities': 'true',
            'limit': 5
        }

        logging.info(f"Sending request to Marketaux API with params: {params}")
        response = requests.get(NEWS_API_URL, params=params)

        if response.status_code == 200:
            data = response.json()
            if 'data' not in data:
                logging.error("No 'data' field in API response.")
                return "Neutral", []

            articles = data.get('data', [])
            sentiment = "Neutral"
            positive_keywords = ['rise', 'bullish', 'strong', 'increase', 'growth', 'positive']
            negative_keywords = ['fall', 'bearish', 'weak', 'decline', 'drop', 'negative']

            positive_count = sum(1 for a in articles if any(word in a.get('title', '').lower() for word in positive_keywords))
            negative_count = sum(1 for a in articles if any(word in a.get('title', '').lower() for word in negative_keywords))

            if positive_count > negative_count:
                sentiment = "Positive"
            elif negative_count > positive_count:
                sentiment = "Negative"

            logging.info(f"Fetched sentiment: {sentiment}")
            return sentiment, articles
        else:
            logging.error(f"Marketaux API request failed with status code: {response.status_code}")
            logging.error(f"Response: {response.text}")
            return "Neutral", []
    except Exception as e:
        logging.error(f"Error fetching sentiment: {e}")
        return "Neutral", []

# Send Email Notification to multiple receivers
def send_email(signal, price, sl, tp):
    try:
        subject = f"Trading Signal: {signal}"
        body = f"Signal: {signal}\nPrice: {price}\nStop Loss: {sl}\nTake Profit: {tp}"

        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = EMAIL_SENDER
        msg['To'] = ", ".join(EMAIL_RECEIVERS)  # Join all receivers with commas

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_RECEIVERS, msg.as_string())  # Send to all receivers

        logging.info(f"Email sent successfully to {len(EMAIL_RECEIVERS)} recipients.")
    except Exception as e:
        logging.error(f"Error sending email: {e}")

# GUI
class TradingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("XAUUSD Trading Bot (3/5 Signals)")
        self.root.geometry("800x700")

        # Signal and Price Labels
        self.label_signal = Label(root, text="Signal: Waiting...", font=("Arial", 18), fg="black")
        self.label_signal.pack(pady=10)

        self.label_price = Label(root, text="Current Price: --", font=("Arial", 14))
        self.label_price.pack()

        self.label_tp = Label(root, text="Take Profit (TP): --", font=("Arial", 14))
        self.label_tp.pack()

        self.label_sl = Label(root, text="Stop Loss (SL): --", font=("Arial", 14))
        self.label_sl.pack()

        # Market Condition Label
        self.label_range_bound = Label(root, text="Market Condition: --", font=("Arial", 14))
        self.label_range_bound.pack()

        # Conditions Met Label (Updated for 3/5)
        self.label_conditions_met = Label(root, text="Conditions Met: BUY (0/5) | SELL (0/5)", font=("Arial", 14))
        self.label_conditions_met.pack(pady=10)

        # Indicator Labels
        self.label_ema_50 = Label(root, text="EMA 50: --", font=("Arial", 12))
        self.label_ema_50.pack()

        self.label_ema_200 = Label(root, text="EMA 200: --", font=("Arial", 12))
        self.label_ema_200.pack()

        self.label_rsi = Label(root, text="RSI: --", font=("Arial", 12))
        self.label_rsi.pack()

        self.label_macd = Label(root, text="MACD: --", font=("Arial", 12))
        self.label_macd.pack()

        self.label_macd_signal = Label(root, text="MACD Signal: --", font=("Arial", 12))
        self.label_macd_signal.pack()

        self.label_atr = Label(root, text="ATR: --", font=("Arial", 12))
        self.label_atr.pack()

        self.label_support = Label(root, text="Support: --", font=("Arial", 12))
        self.label_support.pack()

        self.label_resistance = Label(root, text="Resistance: --", font=("Arial", 12))
        self.label_resistance.pack()

        self.label_volume = Label(root, text="Volume: --", font=("Arial", 12))
        self.label_volume.pack()

        # Sentiment Label
        self.label_sentiment = Label(root, text="Sentiment: Neutral", font=("Arial", 16), fg="blue")
        self.label_sentiment.pack(pady=20)

        # News Label
        self.label_news = Label(root, text="Top News: --", font=("Arial", 12), wraplength=750, justify="left")
        self.label_news.pack()

        # Log Window
        self.log_text = Text(root, height=10, width=95)
        self.log_text.pack(pady=10)
        self.log_text.insert(tk.END, "Logs will appear here...\n")

        # Refresh Button
        self.refresh_button = Button(root, text="Refresh Data", command=self.update_signal_threaded)
        self.refresh_button.pack(pady=10)

        # Start periodic updates
        self.update_signal_threaded()
        self.update_sentiment_threaded()

    def log_message(self, message):
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)

    def update_signal_threaded(self):
        threading.Thread(target=self.update_signal).start()

    def update_signal(self):
        df = get_xauusd_data()
        if not df.empty:
            if is_range_bound(df):
                self.label_range_bound.config(text="Market Condition: Range-Bound", fg="blue")
                self.label_signal.config(text="Signal: WAIT (Range-Bound)", fg="black")
                self.label_price.config(text=f"Current Price: {df['close'].iloc[-1]}")
                self.label_tp.config(text="Take Profit: --")
                self.label_sl.config(text="Stop Loss: --")
                self.label_conditions_met.config(text="Conditions Met: BUY (0/5) | SELL (0/5)")
                self.log_message("Market is range-bound. No signals generated.")
            else:
                self.label_range_bound.config(text="Market Condition: Trending", fg="green")
                signal, sl, tp, buy_conditions_met, sell_conditions_met = generate_signal(df)
                current_price = df['close'].iloc[-1]

                # Update signal and price labels
                self.label_signal.config(text=f"Signal: {signal}", fg="green" if signal == "BUY" else "red" if signal == "SELL" else "black")
                self.label_price.config(text=f"Current Price: {current_price}")
                self.label_tp.config(text=f"Take Profit: {tp}" if tp else "--")
                self.label_sl.config(text=f"Stop Loss: {sl}" if sl else "--")

                # Update conditions met label (3/5)
                self.label_conditions_met.config(text=f"Conditions Met: BUY ({buy_conditions_met}/5) | SELL ({sell_conditions_met}/5)")

                # Update indicator labels
                self.label_ema_50.config(text=f"EMA 50: {df['EMA_50'].iloc[-1]:.2f}")
                self.label_ema_200.config(text=f"EMA 200: {df['EMA_200'].iloc[-1]:.2f}")
                self.label_rsi.config(text=f"RSI: {df['RSI'].iloc[-1]:.2f}")
                self.label_macd.config(text=f"MACD: {df['MACD'].iloc[-1]:.2f}")
                self.label_macd_signal.config(text=f"MACD Signal: {df['MACD_Signal'].iloc[-1]:.2f}")
                self.label_atr.config(text=f"ATR: {df['ATR'].iloc[-1]:.2f}")
                self.label_support.config(text=f"Support: {df['Support'].iloc[-1]:.2f}")
                self.label_resistance.config(text=f"Resistance: {df['Resistance'].iloc[-1]:.2f}")
                self.label_volume.config(text=f"Volume: {df['tick_volume'].iloc[-1]}")

                # Send email if signal generated
                if signal in ["BUY", "SELL"]:
                    send_email(signal, current_price, sl, tp)
                    self.log_message(f"Signal generated and email sent to {len(EMAIL_RECEIVERS)} recipients: {signal} at {current_price}")

        self.root.after(5000, self.update_signal_threaded)

    def update_sentiment_threaded(self):
        threading.Thread(target=self.update_sentiment).start()

    def update_sentiment(self):
        sentiment, articles = fetch_sentiment()
        self.label_sentiment.config(text=f"Sentiment: {sentiment}", fg="green" if sentiment == "Positive" else "red" if sentiment == "Negative" else "blue")
        if articles:
            top_news = articles[0].get('title', 'No news available')
            self.label_news.config(text=f"Top News: {top_news}")
        self.log_message(f"Sentiment updated: {sentiment}")
        self.root.after(60000, self.update_sentiment_threaded)

# Main Application
if __name__ == "__main__":
    root = tk.Tk()
    TradingGUI(root)
    root.mainloop()