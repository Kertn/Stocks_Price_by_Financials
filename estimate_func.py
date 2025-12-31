from sklearn import preprocessing
import numpy as np
import pandas as pd
from statistics import mean
import yfinance as yf
import csv
from tqdm import tqdm

def estimate_annualy_income(model, train_df, y_test, price_discount, bear_inv):
    test_df = train_df.groupby(train_df['Ticker'].astype(int)).last()
    tickers = test_df['Ticker']
    price_coll = test_df['Stock_Price']
    test_df.drop(['Stock_Price', 'Ticker'], axis=1, inplace=True)
    print('Available', len(test_df))
    test_df = preprocessing.normalize(test_df.to_numpy())
    if not len(test_df) == len(tickers) == len(price_coll) == len(y_test):
        print('Len_all', len(test_df), len(tickers), len(price_coll), len(y_test))
        raise ValueError('Len test error')
    total_income_values = []
    total_invested = 0
    total_income = 0
    invest_price = 0
    total_balance = 10000
    invest = False
    for X, ticker, actual_price, y_price in zip(test_df, tickers, price_coll, y_test):

        price_pred = model.predict(X.reshape(1, -1))
        if price_pred * price_discount > actual_price:
            invest = True
            if total_balance < actual_price:
                invest = False
            stocks = int(total_balance / actual_price)
            invest_price = stocks * actual_price
            total_invested += invest_price
            side = 1
        elif (price_pred < actual_price * price_discount) and bear_inv:
            stocks = int(total_balance / actual_price)
            invest_price = stocks * actual_price
            total_invested += invest_price
            side = 0
            invest = True
        else:
            invest_price = 0
            invest = False
        if invest:
            if side:
                total_income += (stocks * y_price) - invest_price
                total_income_values.append(total_income)
            else:
                try:
                    total_income += invest_price - (stocks * y_price)
                    total_income_values.append(total_income)
                except:
                    return 0
        else:
            continue

    print('len_total_invest', len(total_income_values))

    try:
        estim = total_income/total_invested
    except:
        estim = 0
    print(f'Annualy income: {1 + estim:.3f}')
    print("total_income", total_income)
    print("total_invested", total_invested)

    if len(test_df) / 32 > len(total_income_values):
        return 0
    else:
        return estim


def model_invest(model, train_df, y_test, price_discount, bear_inv, sector_name, file_invest_name):
    tickers = train_df['Ticker']
    train_df.drop(['Stock_Price', 'Ticker'], axis=1, inplace=True)
    test_df = preprocessing.normalize(train_df.to_numpy())
    current_income = []
    if not len(test_df) == len(tickers) == len(y_test):
        print('Len_all', len(test_df), len(tickers), len(y_test))
        raise ValueError('Len test error')
    #with open(f'Invest\{file_invest_name}', 'w', newline='') as file:
    # fieldnames = ['Action', 'Ticker', 'Data_Price', 'Predicted_Price', 'Price_Discount', 'CurrentPrice',
    #               'Current_Profit']
    # writer = csv.DictWriter(file, fieldnames=fieldnames)
    # writer.writeheader()
    compan_info = []
    for X, ticker, actual_price in tqdm(zip(test_df, tickers, y_test)):
        current_ticker = str(ticker).split('.')[0]

        price_pred = model.predict(X.reshape(1, -1))

        data = pd.read_csv(fr'All_lists\{sector_name}.csv', encoding='utf-8').values
        ticker_word = data[int(current_ticker) - 1][0]
        company = yf.Ticker(ticker_word)
        fin = company.financials
        all = pd.DataFrame(pd.concat([fin])).columns
        try:
            current_price = company.history(period='5d')['Close'][0]
        except:
            print('1115')
            continue

        #print(f"ticker_word - {ticker_word}, Name - {company.get_info().get('longName')} current_price - {current_price}, price_pred - {price_pred[0]}")

        compan_info.append({
            "Ticker": ticker_word,
            "Name": company.get_info().get("longName"),
            "Current Price": current_price,
            "Predicted Price": price_pred[0],
        })

    df_out = pd.DataFrame(compan_info)

    df_out.to_csv("file_invest_name.csv")

            # if price_pred * price_discount > actual_price:
            #     current_income.append(np.round(((current_price / actual_price) - 1) * 100, 2))
            #     writer.writerow({'Action': 'Buy', 'Ticker': ticker_word, 'Data_Price': actual_price,
            #                          'Predicted_Price': np.round(price_pred, 2),
            #                          'Price_Discount': f'{np.round(((price_pred/actual_price)-1)*100, 2)}%',
            #                          'CurrentPrice': current_price,
            #                          'Current_Profit': f'{np.round(((current_price/actual_price)-1)*100, 2)}%'})
            #
            # elif (price_pred < actual_price * price_discount) and bear_inv:
            #     current_income.append(np.round(((actual_price/current_price)-1)*100, 2))
            #     writer.writerow({'Action': 'Sell', 'Ticker': ticker_word, 'Data_Price': actual_price,
            #                          'Predicted_Price': np.round(price_pred, 2),
            #                          'Price_Discount': f'{np.round(((actual_price/price_pred) - 1) * 100, 2)}%',
            #                          'CurrentPrice': current_price,
            #                          'Current_Profit': f'{np.round(((actual_price/current_price) - 1) * 100, 2)}%'})
        #
        # file.close()
        # print('Current Income', current_income)
        # print('Mean Current Income', mean(current_income), '%')