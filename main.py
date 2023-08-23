from flask import Flask, render_template, request , redirect, url_for
import yfinance as yf
import pandas as pd
import numpy as np
import talib as talib
import itertools 
import plotly.graph_objects as go
import datetime

app = Flask(__name__)

@app.route("/")
def hello_world(name=None):
    
    return render_template('index.html')
    #return render_template('index.html', name=name)




@app.route("/result/")
@app.route('/result/<sonucdegeri>/<tarih>')


def result_world(sonucdegeri="",tarih=0):
    hisseismi= sonucdegeri+" hissesi"
    # Get the stock symbol and data range from the user
    # symbol = input("Enter the stock symbol: ")
    # start_date_str = input("Enter the start date (YYYY-MM-DD): ")

    # Define the date format
   
    symbol = sonucdegeri
    # Get the date input from the form
    start_date_str  =tarih
    # Convert the start date to a datetime object

    # Convert the start date to a datetime object
    date_format = "%Y-%m-%d"
    start_date = datetime.datetime.strptime(start_date_str, date_format).date()

    # Get today's date
    end_date = datetime.date.today()

    # Calculate the number of days
    delta = end_date - start_date
    initial_capital = 1000

    # Retrieve data from Yahoo Finance
    df = yf.download(symbol, start=start_date, end=end_date)

    # Check if the "Close" column containing the closing prices exists
    if "Close" not in df.columns:
        raise ValueError("The dataframe does not contain the 'Close' column.")

    # Convert the dataframe to the appropriate format for calculating technical indicators
    close = df["Close"].values.astype(float)
    high = df["High"].values.astype(float)
    low = df["Low"].values.astype(float)
    open_ = df["Open"].values.astype(float)
    volume = df["Volume"].values.astype(float)

    # Function to generate buy-sell strategies
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
                    macd, _, _ = talib.MACD(close)
                    buy_condition = buy_condition & (macd > 0)
                    sell_condition = sell_condition & (macd < 0)
                elif indicator == "RSI":
                    rsi = talib.RSI(close)
                    buy_condition = buy_condition & (rsi < 35)
                    sell_condition = sell_condition & (rsi > 65)
                elif indicator.startswith("SMA"):
                    period = int(indicator[3:])
                    sma = talib.SMA(close, timeperiod=period)
                    buy_condition = buy_condition & (close > sma)
                    sell_condition = sell_condition & (close < sma)
                elif indicator == "BollingerBands":
                    upper, middle, lower = talib.BBANDS(close)
                    buy_condition = buy_condition & (close > lower)
                    sell_condition = sell_condition & (close < upper)
                elif indicator == "StochasticOscillator":
                    slowk, slowd = talib.STOCH(high, low, close)
                    buy_condition = buy_condition & (slowk < 20) & (slowd < 20)
                    sell_condition = sell_condition & (slowk > 80) & (slowd > 80)
                elif indicator == "ADX":
                    adx = talib.ADX(high, low, close)
                    buy_condition = buy_condition & (adx > 25)
                    sell_condition = sell_condition & (adx < 20)
                elif indicator == "OBV":
                    obv = talib.OBV(close, volume)
                    buy_condition = buy_condition & (obv > 0)
                    sell_condition = sell_condition & (obv < 0)
            strategies.append({
                "strategy": strategy,
                "buy_condition": buy_condition,
                "sell_condition": sell_condition
            })
        return strategies

    # Function to calculate returns for the strategies
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
                    capital=capital*0.998
            elif strategy["sell_condition"][i]:
                if position == 1:
                    position = 0
                    sell_prices.append(close[i])
                    positions.append(position)
                    return_pct = (sell_prices[-1] - buy_prices[-1]) / buy_prices[-1]
                    capital *= (1 + return_pct)
                    capital=capital*0.998
                    returns.append(return_pct)
        if len(buy_prices) > len(sell_prices):
            sell_prices.append(close[-1])
            positions.append(0)
            return_pct = (sell_prices[-1] - buy_prices[-1]) / buy_prices[-1]
            capital *= (1 + return_pct)
            capital=capital*0.998
            returns.append(return_pct)
        return returns, positions, capital

    # Generate strategies
    strategies = generate_strategies(df)

    # Add the buy-and-hold strategy to the list of strategies
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

    # Find the strategy with the highest return
    max_return_index = np.argmax(returns)
    best_strategy = strategies[max_return_index]
    best_returns, best_positions, best_capital = calculate_returns(df, best_strategy, initial_capital)

    # Determine the top 5 strategies with the highest returns
    top_5_strategies = sorted(strategies, key=lambda x: sum(calculate_returns(df, x, initial_capital)[0]), reverse=True)[:5]

    # Get the names and returns of the top 5 strategies
    strategy_names = [f"Strategy {i+1}: {', '.join(strategy['strategy'])}" for i, strategy in enumerate(top_5_strategies)]
    strategy_returns = [sum(calculate_returns(df, strategy, initial_capital)[0]) for strategy in top_5_strategies]

    # Find the index of the best strategy
    best_strategy_index = np.argmax(strategy_returns)
    best_strategy = top_5_strategies[best_strategy_index]
    best_strategy_name = f"Strategy {best_strategy_index + 1}: {', '.join(best_strategy['strategy'])}"
    best_strategy2=best_strategy['strategy']


    # Determine today's strategy
    if best_strategy["buy_condition"][-1] == True:
        today_strategy = "Buy"
    elif best_strategy["sell_condition"][-1] == True:
        today_strategy = "Sell"
    else:
        today_strategy = "Mevcut Pozisyonu Koru"

    # If the best strategy is the buy-and-hold strategy, set today's strategy to "Hold"
    if best_strategy["strategy"] == ("Buy and Hold",):
        today_strategy = "Hold"

    # If the best strategy is the buy-and-hold strategy, set today's strategy to "Hold"
    if best_strategy["strategy"] == ("Buy and Hold",):
        today_strategy = "Hold"

    # Show the returns and today's strategy
    import plotly.graph_objects as go

    # Determine the top 5 strategies with the highest returns
    top_5_strategies = sorted(strategies, key=lambda x: sum(calculate_returns(df, x, initial_capital)[0]), reverse=True)[:5]

    # Get the names and returns of the top 5 strategies
    strategy_names = [f"Strategy {i+1}: {strategy['strategy']}" for i, strategy in enumerate(top_5_strategies)]
    strategy_returns = [sum(calculate_returns(df, strategy, initial_capital)[0]) for strategy in top_5_strategies]

    # Create a visualization
    fig = go.Figure(data=[go.Bar(x=strategy_names, y=strategy_returns)])
    fig.update_layout(title='Top 5 Strategies and Returns',
                    xaxis_title='Strategies',
                    yaxis_title='Return Rate')
    #fig.show()
    print("\nStrategy with the Highest Return out of 512 Strategies:")
    print(f"Strategy: {best_strategy['strategy']} - Return Rate: {sum(best_returns):.4f} - \nInitial Capital of {initial_capital} Turkish Liras has become {best_capital:.4f} Turkish Liras in {delta.days} days since {start_date_str}")
    print("\nToday's Strategy:",{best_strategy['strategy']})
    print(f"{today_strategy}")

    # Pass the data to the 'result.html' template
    return render_template('result.html', sonuc=hisseismi, today=today_strategy,
                        initial_capital=initial_capital,
                       start_date=start_date_str, final_capital=best_capital,best_strategy_name=best_strategy2)




if __name__ == '__main__':
    app.run(host='0.0.0.0' debug=False)





