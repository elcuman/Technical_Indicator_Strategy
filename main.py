from flask import Flask, render_template, request, redirect, url_for
import yfinance as yf
import pandas as pd
import numpy as np
import itertools
import plotly.graph_objects as go
import datetime
from datetime import timedelta



app = Flask(__name__)

@app.route("/")
def hello_world(name=None):
    return render_template('index.html')


@app.route('/result/<sonucdegeri>/<tarih>')
def result_world(sonucdegeri="", tarih=0):
    hisseismi = sonucdegeri + " hissesi"

    symbol = sonucdegeri
    start_date_str = tarih

    date_format = "%Y-%m-%d"
    start_date = datetime.datetime.strptime(start_date_str, date_format).date()
    end_date = datetime.date.today()
    delta = end_date - start_date
    initial_capital = 1000

    df = yf.download(symbol, start=start_date, end=end_date)

    if "Close" not in df.columns:
        raise ValueError("The dataframe does not contain the 'Close' column.")

    close = df["Close"].values.astype(float)
    high = df["High"].values.astype(float)
    low = df["Low"].values.astype(float)
    open_ = df["Open"].values.astype(float)
    volume = df["Volume"].values.astype(float)
    # Define the periods for the indicators
    sma9 = 9
    sma21 = 21
    sma50 = 50
    rsi_period = 14
    stoch_period = 14
    adx_period = 14

    # Calculate Simple Moving Averages (SMA)
    df['SMA9'] = df['Close'].rolling(window=sma9).mean()
    df['SMA21'] = df['Close'].rolling(window=sma21).mean()
    df['SMA50'] = df['Close'].rolling(window=sma50).mean()

    # Calculate Relative Strength Index (RSI)
    delta = df['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=rsi_period).mean()
    avg_loss = loss.rolling(window=rsi_period).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Calculate Bollinger Bands
    bb_period = 20  # Bollinger Bands period
    rolling_mean = df['Close'].rolling(window=bb_period).mean()
    rolling_std = df['Close'].rolling(window=bb_period).std()
    df['BollingerBands_Mid'] = rolling_mean
    df['BollingerBands_Up'] = rolling_mean + (rolling_std * 2)
    df['BollingerBands_Down'] = rolling_mean - (rolling_std * 2)

    # Calculate Stochastic Oscillator
    high_max = df['High'].rolling(window=stoch_period).max()
    low_min = df['Low'].rolling(window=stoch_period).min()
    df['StochasticOscillator'] = ((df['Close'] - low_min) / (high_max - low_min)) * 100

    # Calculate Average Directional Index (ADX)
    def calculate_dx(df):
        df['High-Low'] = df['High'] - df['Low']
        df['High-Close'] = abs(df['High'] - df['Close'].shift())
        df['Low-Close'] = abs(df['Low'] - df['Close'].shift())
        df['TR'] = df[['High-Low', 'High-Close', 'Low-Close']].max(axis=1)
        df['+DM'] = np.where((df['High-Close'] > df['Low-Close']) & (df['High-Close'] > 0), df['High-Close'], 0)
        df['-DM'] = np.where((df['Low-Close'] > df['High-Close']) & (df['Low-Close'] > 0), df['Low-Close'], 0)
        df['+DI'] = (df['+DM'].rolling(window=adx_period).sum() / df['TR'].rolling(window=adx_period).sum()) * 100
        df['-DI'] = (df['-DM'].rolling(window=adx_period).sum() / df['TR'].rolling(window=adx_period).sum()) * 100
        df['DX'] = (abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])) * 100
        return df

    df = calculate_dx(df)
    df['ADX'] = df['DX'].rolling(window=adx_period).mean()

    # Calculate On-Balance Volume (OBV)
    df['OBV'] = np.where(df['Close'] > df['Close'].shift(), df['Volume'], -df['Volume'])
    df['OBV'] = df['OBV'].cumsum()

    # Calculate MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26



    def generate_strategies(df):
        strategies = []
        indicators = ["MACD", "RSI", "SMA9", "SMA21", "SMA50", "BollingerBands", "StochasticOscillator", "ADX", "OBV"]
        strategy_names = []
        for i in range(1, len(indicators) + 1):
            strategy_names += list(itertools.combinations(indicators, i))
        for strategy in strategy_names:
            buy_condition = np.full_like(close, True, dtype=bool)
            sell_condition = np.full_like(close, True, dtype=bool)
            for indicator in strategy:
                if indicator == "MACD":
                    macd = df['MACD']
                    buy_condition = buy_condition & (macd > 0)
                    sell_condition = sell_condition & (macd < 0)
                elif indicator == "RSI":
                    rsi = df['RSI']
                    buy_condition = buy_condition & (rsi < 35)
                    sell_condition = sell_condition & (rsi > 65)
                elif indicator.startswith("SMA"):
                    period = int(indicator[3:])
                    sma = df[f'SMA{period}']
                    buy_condition = buy_condition & (close > sma)
                    sell_condition = sell_condition & (close < sma)
                elif indicator == "BollingerBands":
                    lower = df['BollingerBands_Down']
                    upper = df['BollingerBands_Up']
                    buy_condition = buy_condition & (close > lower)
                    sell_condition = sell_condition & (close < upper)
                elif indicator == "StochasticOscillator":
                    stoch = df['StochasticOscillator']
                    buy_condition = buy_condition & (stoch < 20)
                    sell_condition = sell_condition & (stoch > 80)
                elif indicator == "ADX":
                    adx = df['ADX']
                    buy_condition = buy_condition & (adx > 25)
                    sell_condition = sell_condition & (adx < 20)
                elif indicator == "OBV":
                    obv = df['OBV']
                    buy_condition = buy_condition & (obv > 0)
                    sell_condition = sell_condition & (obv < 0)
            strategies.append({
                "strategy": strategy,
                "buy_condition": buy_condition,
                "sell_condition": sell_condition
            })
        return strategies



    def calculate_returns(df, strategy, initial_capital):
        returns = []
        buy_prices = []
        sell_prices = []
        positions = []
        position = 0
        capital = initial_capital
        for i in range(len(df)):
            if strategy["buy_condition"][i]:
                if position == 0:
                    position = 1
                    buy_prices.append(close[i])
                    positions.append(position)
                    capital = capital * 0.998
            elif strategy["sell_condition"][i]:
                if position == 1:
                    position = 0
                    sell_prices.append(close[i])
                    positions.append(position)
                    return_pct = (sell_prices[-1] - buy_prices[-1]) / buy_prices[-1]
                    capital *= (1 + return_pct)
                    capital = capital * 0.998
                    returns.append(return_pct)
        if len(buy_prices) > len(sell_prices):
            sell_prices.append(close[-1])
            positions.append(0)
            return_pct = (sell_prices[-1] - buy_prices[-1]) / buy_prices[-1]
            capital *= (1 + return_pct)
            capital = capital * 0.998
            returns.append(return_pct)
        return returns, positions, capital

    strategies = generate_strategies(df)

    strategies.append({
        "strategy": ("Buy and Hold",),
        "buy_condition": np.full_like(close, True, dtype=bool),
        "sell_condition": np.full_like(close, False, dtype=bool)
    })

    returns = []
    capitals = []
    for strategy in strategies:
        strategy_returns, _, strategy_capital = calculate_returns(df, strategy, initial_capital)
        returns.append(sum(strategy_returns))
        capitals.append(strategy_capital)

    max_return_index = np.argmax(returns)
    best_strategy = strategies[max_return_index]
    best_returns, best_positions, best_capital = calculate_returns(df, best_strategy, initial_capital)

    top_5_strategies = sorted(strategies, key=lambda x: sum(calculate_returns(df, x, initial_capital)[0]), reverse=True)[:5]

    strategy_names = [f"Strategy {i + 1}: {', '.join(strategy['strategy'])}" for i, strategy in enumerate(top_5_strategies)]
    strategy_returns = [sum(calculate_returns(df, strategy, initial_capital)[0]) for strategy in top_5_strategies]

    best_strategy_index = np.argmax(strategy_returns)
    best_strategy = top_5_strategies[best_strategy_index]
    best_strategy_name = f"Strategy {best_strategy_index + 1}: {', '.join(best_strategy['strategy'])}"
    best_strategy2 = best_strategy['strategy']

    if best_strategy["buy_condition"][-1] == True:
        today_strategy = "Buy"
    elif best_strategy["sell_condition"][-1] == True:
        today_strategy = "Sell"
    else:
        today_strategy = "Mevcut Pozisyonu Koru"

    if best_strategy["strategy"] == ("Buy and Hold",):
        today_strategy = "Hold"

    if best_strategy["strategy"] == ("Buy and Hold",):
        today_strategy = "Hold"

    top_5_strategies = sorted(strategies, key=lambda x: sum(calculate_returns(df, x, initial_capital)[0]), reverse=True)[:5]

    strategy_names = [f"Strategy {i + 1}: {strategy['strategy']}" for i, strategy in enumerate(top_5_strategies)]
    strategy_returns = [sum(calculate_returns(df, strategy, initial_capital)[0]) for strategy in top_5_strategies]


    return render_template('result.html', sonuc=hisseismi, today=today_strategy,
                           initial_capital=initial_capital,
                           start_date=start_date_str, final_capital=best_capital,
                           best_strategy_name=best_strategy2,
                           top_5_strategies=top_5_strategies, strategy_returns=strategy_returns)


if __name__ == '__main__':
    app.run()
