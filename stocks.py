import yfinance as yf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
import random
import pandas as pd
import datetime
import math
import os

np.random.seed(123)
tf.random.set_seed(123)
random.seed(123)

def get_stock_list():
    # return ['ACGL', 'ACWX', 'AEM', 'AER', 
    #         'AFL', 'AIG', 'ARKB', 'ASO', 'BALL', 'BBJP', 'BBY', 'BIL', 
    #         'BJ', 'BRBR', 'BRO', 'C', 'CARR', 'CBRE', 'CCEP', 'CF', 'CHK', 'CIVI',
    #         'CL', 'CMC', 'CNM', 'CNQ', 'COWZ', 'CP', 'CPRT', 'CRH', 'CSGP', 'CTVA', 'CVNA', 'CVS',
    #         'DD', 'DGRO', 'DINO', 'DOCU', 'DOW', 'EBAY', 'EFA', 'EFV', 'EIX', 'EMB', 'EMN', 'EMXC', 'EQR', 'EW', 'EWBC', 
    #         'EWJ', 'EWW', 'EWY', 'EZU', 'FAST', 'FBTC', 'FEZ', 'FIS', 'FLOT', 'FTV', 'GBTC', 'HIG', 'HOLX', 'HWM', 'IEFA', 'IEMG',
    #         'IFF', 'IJH', 'IR', 'IRM', 'IVW', 'IWR', 'IXUS', 'JAAA', 'JCI',
    #         'AA', 'AAAU', 'AEO', 'AGQ', 'ALLY', 'ALPN', 'AMH', 'AMLP', 'APG', 'AR',
    #         'ATI', 'ATMU', 'AU', 'AVTR', 'AXTA', 'AZEK', 'BAC', 'BITB', 'BKLN', 'BLMN',
    #         'BNS', 'BOTZ', 'BOX', 'BP', 'BROS', 'BSCO', 'BUFR', 'BXSL', 'CART', 'CFG',
    #         'CGDV', 'CGGR', 'CHX', 'CM', 'CNX', 'CRBG', 'CTRA', 'CTRE', 'D', 'DAL', 'DFAC', 'DFAI', 'DFIC', 'DOCN', 'DVN', 'DYN', 'DYNF',
    #         'EAT', 'EDR', 'EEM', 'ENB', 'EPD', 'EPRT', 'EQH', 'ERJ', 'ESI', 'EVBG', 'EWC', 'EWG', 'EWT', 'EWU', 'EXEL', 'FCX', 
    #         'FDMT', 'FE', 'FITB', 'FLR', 'FNDF', 'FRO', 'FTI', 'FVD', 'FYBR', 'GCT']
    return ['AAP', 'ACGL']

def get_stock_data(stock_ticker, days_back, end_date):
    start_date = end_date - datetime.timedelta(days=days_back)
    stock_data, sp500_data, nasdaq_data, djia_data, volume_data = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    try:
        stock_data = yf.download(stock_ticker, start=start_date, end=end_date)
        sp500_data = yf.download('^GSPC', start=start_date, end=end_date)['Close']  # S&P 500
        nasdaq_data = yf.download('^IXIC', start=start_date, end=end_date)['Close']  # NASDAQ
        djia_data = yf.download('^DJI', start=start_date, end=end_date)['Close']  # Dow Jones Industrial Average
        volume_data = stock_data['Volume']
        stock_data = stock_data['Close']
    except Exception as e:
        print(f"Error downloading data: {e}")
    
    return stock_data, sp500_data, nasdaq_data, djia_data, volume_data


def preprocess_data(stock_data, sp500_data, nasdaq_data, djia_data, volume_data, sequence_length):
    scaler_stock = MinMaxScaler(feature_range=(0, 1))
    scaled_stock_data = scaler_stock.fit_transform(stock_data.values.reshape(-1, 1))
    
    scaler_sp500 = MinMaxScaler(feature_range=(0, 1))
    scaled_sp500_data = scaler_sp500.fit_transform(sp500_data.values.reshape(-1, 1))
    
    scaler_nasdaq = MinMaxScaler(feature_range=(0, 1))
    scaled_nasdaq_data = scaler_nasdaq.fit_transform(nasdaq_data.values.reshape(-1, 1))
    
    scaler_djia = MinMaxScaler(feature_range=(0, 1))
    scaled_djia_data = scaler_djia.fit_transform(djia_data.values.reshape(-1, 1))

    scaler_volume = MinMaxScaler(feature_range=(0, 1))
    scaled_volume_data = scaler_volume.fit_transform(volume_data.values.reshape(-1, 1))

    X, Y = [], []
    for i in range(len(scaled_stock_data) - sequence_length):
        X.append(np.hstack((scaled_stock_data[i:i + sequence_length], 
                            scaled_sp500_data[i:i + sequence_length], 
                            scaled_nasdaq_data[i:i + sequence_length],
                            scaled_djia_data[i:i + sequence_length],
                            scaled_volume_data[i:i + sequence_length])))  # Include volume and DJIA data
        Y.append(scaled_stock_data[i + sequence_length])

    return np.array(X), np.array(Y), scaler_stock


def build_lstm_model(input_shape, units=50, lstm_layers=2, dropout_rate=0.2):
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=lstm_layers > 1, input_shape=(input_shape[0], 5)))
    model.add(Dropout(dropout_rate))

    for i in range(1, lstm_layers):
        model.add(LSTM(units=units, return_sequences=i < lstm_layers - 1))
        model.add(Dropout(dropout_rate))

    model.add(Dense(units=units // 2, activation='relu'))  # Additional Dense layer
    model.add(Dropout(dropout_rate))  # Additional Dropout layer
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model




def evaluate_model(model, X_test, Y_test, scaler, average_price):
    predicted = model.predict(X_test)
    predicted = scaler.inverse_transform(predicted)
    real = scaler.inverse_transform(Y_test.reshape(-1, 1))

    # Calculate weights - more recent dates have exponentially higher weights
    num_points = len(Y_test)
    weights = np.exp(np.linspace(0, 5, num_points))  # Exponential increase in weights
    weights /= np.sum(weights)  # Normalize weights to sum to 1

    # Calculate weighted absolute error and normalize by the number of points
    wae = np.sum(weights * np.abs(real - predicted)) / num_points
    wnmae = (wae / average_price) * 100  # Weighted NMAE as a percentage
    return wnmae


def predict_next_day_close(model, last_sequence, scaler):
    last_sequence = last_sequence.reshape((1, last_sequence.shape[0], last_sequence.shape[1]))
    predicted = model.predict(last_sequence)
    predicted = scaler.inverse_transform(predicted)
    return predicted[0, 0]


def analyze_stocks(today_date, budget=1900):
    seeds = [123, 124, 125]
    stock_list = get_stock_list()
    days_back = 1000
    sequence_length = 60
    lstm_units = 100
    lstm_layers = 3
    dropout_rate = 0.3
    aggregated_results = {}

    for seed in seeds:
        np.random.seed(seed)
        tf.random.set_seed(seed)
        random.seed(seed)

        for stock_ticker in stock_list:
            end_date = today_date + datetime.timedelta(days=1)
            stock_data, sp500_data, nasdaq_data, djia_data, volume_data = get_stock_data(stock_ticker, days_back, end_date)

            if stock_data.empty or sp500_data.empty or nasdaq_data.empty or djia_data.empty or volume_data.empty:
                print(f"Insufficient data for {stock_ticker}")
                continue

            average_price = stock_data.mean()
            X, Y, scaler = preprocess_data(stock_data[:-1], sp500_data[:-1], nasdaq_data[:-1], djia_data[:-1], volume_data[:-1], sequence_length)
            split = int(0.8 * len(X))

            if len(X) <= 1 or len(X[0]) != sequence_length:
                continue

            X_train, Y_train, X_test, Y_test = X[:split], Y[:split], X[split:], Y[split:]

            if X_train.shape[0] < 1:
                continue

            model = build_lstm_model(X_train.shape[1:], units=lstm_units, lstm_layers=lstm_layers, dropout_rate=dropout_rate)
            model.fit(X_train, Y_train, epochs=50, batch_size=32)

            nmae = evaluate_model(model, X_test, Y_test, scaler, average_price)
            todays_close = stock_data.iloc[-1]
            next_day_close = predict_next_day_close(model, X[-1], scaler)

            if stock_ticker not in aggregated_results:
                aggregated_results[stock_ticker] = {'nmae': [], 'todays_close': [], 'predicted_close': [], 'investment_cost': [], 'projected_profit_loss': [], 'positive_prediction_count': 0, 'avg_shares_to_buy': 0, 'avg_investment_cost': 0}

            aggregated_results[stock_ticker]['nmae'].append(nmae)
            aggregated_results[stock_ticker]['todays_close'].append(todays_close)
            aggregated_results[stock_ticker]['predicted_close'].append(next_day_close)
            if next_day_close > todays_close:
                aggregated_results[stock_ticker]['positive_prediction_count'] += 1

    # Pre-calculation of the total investment cost and shares to buy for each stock
    total_investment_cost = 0
    for stock_ticker, results in aggregated_results.items():
        if len(results['nmae']) > 0:
            avg_todays_close = sum(results['todays_close']) / len(results['todays_close'])
            avg_nmae = sum(results['nmae']) / len(results['nmae'])
            avg_predicted_close = sum(results['predicted_close']) / len(results['predicted_close'])

            predicted_change_ratio = ((avg_predicted_close - avg_todays_close) / avg_todays_close) * 100 / avg_nmae
            
            ### FEEL FREE TO CHANGE ###
            avg_shares_to_buy = 0
            if predicted_change_ratio > 2.5:
                avg_shares_to_buy = 60
            elif predicted_change_ratio > 2:
                avg_shares_to_buy = 36
            elif predicted_change_ratio > 1.5:
                avg_shares_to_buy = 12
            elif predicted_change_ratio > 1.25:
                avg_shares_to_buy = 4
            elif predicted_change_ratio > 1:
                avg_shares_to_buy = 2
            ### FEEL FREE TO CHANGE ###

            results['avg_shares_to_buy'] = avg_shares_to_buy
            results['avg_investment_cost'] = avg_todays_close * avg_shares_to_buy
            total_investment_cost += results['avg_investment_cost']
        else:
            print(f"No valid NMAE values for {stock_ticker}, skipping average calculations.")
            continue

    # Adjust shares to buy if over budget
    if total_investment_cost > budget:
        over_budget_ratio = budget / total_investment_cost
        for stock_ticker, results in aggregated_results.items():
            adjusted_shares_to_buy = math.floor(results['avg_shares_to_buy'] * over_budget_ratio)
            results['avg_shares_to_buy'] = adjusted_shares_to_buy
            results['avg_investment_cost'] = results['todays_close'][0] * adjusted_shares_to_buy

    # Recalculate the total investment cost after adjustment
    total_investment_cost = sum([results['avg_investment_cost'] for stock_ticker, results in aggregated_results.items()])

    # Calculate the projected profit/loss based on adjusted shares
    total_avg_projected_profit_loss = 0
    output_lines = []  # Initialize a list to hold the lines of text to be written to the file

    for stock_ticker, results in aggregated_results.items():
        if len(results['nmae']) > 0:
            avg_nmae = sum(results['nmae']) / len(results['nmae'])
            avg_todays_close = sum(results['todays_close']) / len(results['todays_close'])
            avg_predicted_close = sum(results['predicted_close']) / len(results['predicted_close'])
            avg_shares_to_buy = results['avg_shares_to_buy']
            avg_investment_cost = results['avg_investment_cost']
            avg_projected_profit_loss = (avg_predicted_close - avg_todays_close) * avg_shares_to_buy
            results['avg_projected_profit_loss'] = avg_projected_profit_loss

            total_avg_projected_profit_loss += avg_projected_profit_loss

            # Append formatted string to output_lines
            output_lines.append(f"Stock: {stock_ticker}, NMAE: {round(avg_nmae, 4)}, Today's Close: {round(avg_todays_close, 4)}, "
                                f"Predicted Next Day Close: {round(avg_predicted_close, 4)}, Avg Shares to Buy: {avg_shares_to_buy}, "
                                f"Avg Investment Cost: {round(avg_investment_cost, 4)}, Avg Projected Profit/Loss: {round(avg_projected_profit_loss, 4)}\n")
        else:
            # Handle stocks with no valid NMAE values
            output_lines.append(f"No valid NMAE values for {stock_ticker}, skipping final calculations.\n")
            continue

    # Append summary lines
    output_lines.append(f"Total Avg Investment Cost: {round(total_investment_cost, 4)}\n")
    output_lines.append(f"Total Avg Projected Profit/Loss: {round(total_avg_projected_profit_loss, 4)}\n")
    total_avg_percentage_change = (total_avg_projected_profit_loss / total_investment_cost) * 100 if total_investment_cost != 0 else 0
    output_lines.append(f"Total Avg Projected Percentage Change: {round(total_avg_percentage_change, 2)}%\n")

    # Specify the directory and filename
    directory = "analyze"
    os.makedirs(directory, exist_ok=True)  # Ensure the directory exists
    file_name = f"{directory}/analyze_results_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.txt"

    # Write the output lines to a file
    with open(file_name, 'w') as file:
        file.writelines(output_lines)




def simulate_stocks(today_date, tomorrow_date, budget=1300):
    seeds = [123, 124, 125]
    stock_list = get_stock_list()
    days_back = 1000
    sequence_length = 60
    lstm_units = 100
    lstm_layers = 3
    dropout_rate = 0.3
    aggregated_results = {}

    for stock_ticker in stock_list:
        aggregated_results[stock_ticker] = {'nmae': [], 'todays_close': [], 'predicted_close': [], 'predicted_next_day_close': [], 'investment_cost': [], 'profit_loss': [], 'positive_prediction_count': 0, 'avg_shares_to_buy': 0, 'avg_investment_cost': 0}

    for seed in seeds:
        np.random.seed(seed)
        tf.random.set_seed(seed)
        random.seed(seed)

        for stock_ticker in stock_list:
            day_after_tomorrow = tomorrow_date + datetime.timedelta(days=1)
            stock_data, sp500_data, nasdaq_data, djia_data, volume_data = get_stock_data(stock_ticker, days_back, day_after_tomorrow)

            if stock_data.empty or sp500_data.empty or nasdaq_data.empty or djia_data.empty or volume_data.empty:
                print(f"Insufficient data for {stock_ticker}")
                continue

            todays_close = stock_data.loc[today_date.strftime('%Y-%m-%d')] if today_date.strftime('%Y-%m-%d') in stock_data.index else None
            actual_next_day_close = stock_data.loc[tomorrow_date.strftime('%Y-%m-%d')] if tomorrow_date.strftime('%Y-%m-%d') in stock_data.index else None

            if todays_close is None or actual_next_day_close is None:
                print(f"Data not available for {stock_ticker} on specified dates.")
                continue

            aggregated_results[stock_ticker]['todays_close'].append(todays_close)
            aggregated_results[stock_ticker]['predicted_close'].append(actual_next_day_close)

            average_price = stock_data.mean()
            X, Y, scaler = preprocess_data(stock_data.iloc[:-1], sp500_data.iloc[:-1], nasdaq_data.iloc[:-1], djia_data.iloc[:-1], volume_data.iloc[:-1], sequence_length)

            split = int(0.8 * len(X))

            if len(X) <= 1 or len(X[0]) != sequence_length:
                continue

            X_train, Y_train, X_test, Y_test = X[:split], Y[:split], X[split:], Y[split:]

            if X_train.shape[0] < 1:
                continue

            model = build_lstm_model(X_train.shape[1:], units=lstm_units, lstm_layers=lstm_layers, dropout_rate=dropout_rate)
            model.fit(X_train, Y_train, epochs=50, batch_size=32)

            nmae = evaluate_model(model, X_test, Y_test, scaler, average_price)
            predicted_next_day_close = predict_next_day_close(model, X[-1], scaler)

            aggregated_results[stock_ticker]['nmae'].append(nmae)
            aggregated_results[stock_ticker]['predicted_next_day_close'].append(predicted_next_day_close)
            if predicted_next_day_close > todays_close:
                aggregated_results[stock_ticker]['positive_prediction_count'] += 1

    # Pre-calculation of the total investment cost and shares to buy for each stock
    total_investment_cost = 0
    for stock_ticker, results in aggregated_results.items():
        if len(results['nmae']) > 0:
            avg_nmae = sum(results['nmae']) / len(results['nmae']) if len(results['nmae']) > 0 else 0
            avg_todays_close = sum(results['todays_close']) / len(results['todays_close'])
            avg_predicted_next_day_close = sum(results['predicted_next_day_close']) / len(results['predicted_next_day_close'])

            predicted_change_ratio = (avg_predicted_next_day_close - avg_todays_close) * 100 / avg_todays_close / avg_nmae

            ### FEEL FREE TO CHANGE ###
            avg_shares_to_buy = 0
            if predicted_change_ratio > 2.5:
                avg_shares_to_buy = 15
            elif predicted_change_ratio > 2:
                avg_shares_to_buy = 9
            elif predicted_change_ratio > 1.5:
                avg_shares_to_buy = 3
            elif predicted_change_ratio > 1.25:
                avg_shares_to_buy = 1
            elif predicted_change_ratio > 1:
                avg_shares_to_buy = 2
            ### FEEL FREE TO CHANGE ###

            results['avg_shares_to_buy'] = avg_shares_to_buy  # Add to results for each stock ticker
            results['avg_investment_cost'] = avg_todays_close * avg_shares_to_buy  # Set avg_investment_cost
            total_investment_cost += results['avg_investment_cost']
        else:
            print(f"No valid NMAE values for {stock_ticker}, skipping average calculations.")
            continue

    # Adjust shares to buy if over budget
    if total_investment_cost > budget:
        over_budget_ratio = budget / total_investment_cost
        for stock_ticker, results in aggregated_results.items():
            adjusted_shares_to_buy = math.floor(results['avg_shares_to_buy'] * over_budget_ratio)
            results['avg_shares_to_buy'] = adjusted_shares_to_buy  # Update the number of shares to buy
            results['avg_investment_cost'] = results['todays_close'][0] * adjusted_shares_to_buy

    # Recalculate the total investment cost after adjustment
    total_investment_cost = sum([results['avg_investment_cost'] for stock_ticker, results in aggregated_results.items()])

    # Calculate the profit/loss based on the actual next day close and adjusted shares
    # Initialize a list to hold the lines of text to be written to the file
    output_lines = []

    total_avg_profit_loss = 0
    for stock_ticker, results in aggregated_results.items():
        if len(results['nmae']) > 0:
            avg_nmae = sum(results['nmae']) / len(results['nmae'])
            avg_todays_close = sum(results['todays_close']) / len(results['todays_close'])
            avg_predicted_next_day_close = sum(results['predicted_next_day_close']) / len(results['predicted_next_day_close'])
            avg_actual_next_day_close = sum(results['predicted_close']) / len(results['predicted_close'])  # This line correctly calculates the actual next day close
            avg_shares_to_buy = results['avg_shares_to_buy']
            avg_investment_cost = results['avg_investment_cost']
            avg_profit_loss = (avg_actual_next_day_close - avg_todays_close) * avg_shares_to_buy
            results['avg_profit_loss'] = avg_profit_loss

            total_avg_profit_loss += avg_profit_loss

            # Append formatted string to output_lines instead of printing
            output_lines.append(f"Stock: {stock_ticker}, NMAE: {round(avg_nmae, 4)}, Today's Close: {round(avg_todays_close, 4)}, "
                    f"Predicted Next Day Close: {round(avg_predicted_next_day_close, 4)}, Actual Next Day Close: {round(avg_actual_next_day_close, 4)}, "
                    f"Avg Shares to Buy: {avg_shares_to_buy}, Avg Investment Cost: {round(avg_investment_cost, 4)}, "
                    f"Avg Profit/Loss: {round(avg_profit_loss, 4)}\n")
        else:
            # Handle stocks with no valid NMAE values
            output_lines.append(f"No valid NMAE values for {stock_ticker}, skipping final calculations.\n")
            continue

    # Append summary lines based on total investment cost condition
    if total_investment_cost > 0:
        output_lines.append(f"Total Avg Investment: {round(total_investment_cost, 4)}\n")
        output_lines.append(f"Total Avg Profit/Loss: {round(total_avg_profit_loss, 4)}\n")
        total_avg_percentage_change = (total_avg_profit_loss / total_investment_cost) * 100
        output_lines.append(f"Total Avg Percentage Change: {round(total_avg_percentage_change, 2)}%\n")
    else:
        output_lines.append("No valid investment data available.\n")

    # Specify the directory and filename
    directory = "simulate"
    os.makedirs(directory, exist_ok=True)  # Ensure the directory exists
    file_name = f"{directory}/simulate_results_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.txt"

    # Write the output lines to a file
    with open(file_name, 'w') as file:
        file.writelines(output_lines)

















def main():
    choice = input("Type 1 for analyze, 2 for simulate: ")

    if choice == "1":
        input_date_str = input("Enter today's date in MMDDYYYY format: ")
        try:
            today_date = datetime.datetime.strptime(input_date_str, '%m%d%Y')
        except ValueError:
            print("Invalid date format. Please enter the date in MMDDYYYY format.")
            return

        analyze_stocks(today_date)

    elif choice == "2":
        input_today_str = input("Enter today's date in MMDDYYYY format: ")
        input_tomorrow_str = input("Enter tomorrow's date in MMDDYYYY format: ")

        try:
            today_date = datetime.datetime.strptime(input_today_str, '%m%d%Y')
            tomorrow_date = datetime.datetime.strptime(input_tomorrow_str, '%m%d%Y')
        except ValueError:
            print("Invalid date format. Please enter the date in MMDDYYYY format.")
            return

        if today_date >= tomorrow_date:
            print("Tomorrow's date must be after today's date.")
            return

        simulate_stocks(today_date, tomorrow_date)
    else:
        print("Invalid choice. Please enter 1 or 2.")

main()

