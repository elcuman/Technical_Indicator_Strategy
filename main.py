from flask import Flask, render_template, request, redirect, url_for
import yfinance as yf
import pandas as pd
import numpy as np
import itertools
import plotly.graph_objects as go
import datetime

app = Flask(__name__)

@app.route("/")
def hello_world(name=None):
    return render_template('index.html')

@app.route("/result/")
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
                    macd = (close - np.mean(close)) / np.std(close)
                    buy_condition = buy_condition & (macd > 0)
                    sell_condition = sell_condition & (macd < 0)
                elif indicator == "RSI":
                    rsi = 100 - (100 / (1 + (np.mean(close[:14]) / np.mean(close[14:]) if np.mean(close[14:]) != 0 else 1)))
                    buy_condition = buy_condition & (rsi < 35)
                    sell_condition = sell_condition & (rsi > 65)
                elif indicator.startswith("SMA"):
                    period = int(indicator[3:])
                    sma = np.mean(close[-period:])
                    buy_condition = buy_condition & (close > sma)
                    sell_condition = sell_condition & (close < sma)
                elif indicator == "BollingerBands":
                    sma = np.mean(close)
                    std = np.std(close)
                    upper = sma + 2 * std
                    lower = sma - 2 * std
                    buy_condition = buy_condition & (close > lower)
                    sell_condition = sell_condition & (close < upper)
                elif indicator == "StochasticOscillator":
                    k_period = 14
                    d_period = 3
                    k_values = 100 * ((close - np.min(low[-k_period:])) / (np.max(high[-k_period:]) - np.min(low[-k_period:])))
                    d_values = np.mean(k_values[-d_period:])
                    buy_condition = buy_condition & (k_values[-1] < 20) & (d_values < 20)
                    sell_condition = sell_condition & (k_values[-1] > 80) & (d_values > 80)
                elif indicator == "ADX":
                    adx_period = 14
                    dx = 100 * np.mean(np.abs((np.mean(high[-adx_period:]) - np.mean(low[-adx_period:])) / (np.mean(high[-adx_period:]) + np.mean(low[-adx_period:]))))
                    buy_condition = buy_condition & (dx > 25)
                    sell_condition = sell_condition & (dx < 20)
                elif indicator == "OBV":
                    obv_values = np.where(close > close[-1], volume, -volume)
                    obv = np.cumsum(obv_values)
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

    fig = go.Figure(data=[go.Bar(x=strategy_names, y=strategy_returns)])
    fig.update_layout(title='Top 5 Strategies and Returns',
                      xaxis_title='Strategies',
                      yaxis_title='Return Rate')

    return render_template('result.html', sonuc=hisseismi, today=today_strategy,
                           initial_capital=initial_capital,
                           start_date=start_date_str, final_capital=best_capital, best_strategy_name=best_strategy2)

if __name__ == '__main__':
    app.run()
