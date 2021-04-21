import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error as mse


prediction_length = 10
selected_stock_sbl = ""
prediction_accuracy = 0

# define dictionary with current offered stocks and stock symbols
avb_stock = {
    "APPL": "Apple Inc",
    "ADBE": "Adobe Inc",
    "AMZN": "Amazon.com Inc",
    "BAC": "Bank of America Corp",
    "CRM": "Salesforce.com inc",
    "DIS": "Walt Disney Co",
    "FB": "Facebook Inc",
    "GOOGL": "Alphabet Icn",
    "HD": "Home Depot Inc",
    "INTC": "Intel Corporation",
    "JNJ": "Johnson & Johnson",
    "JPM": "JPMorgan Chase & Co",
    "MA": "Mastercard Inc",
    "NFLX": "Netflix Inc",
    "NVDA": "NVIDIA Corporation",
    "PYPL": "Paypal Holdings Inc",
    "TSLA": "Tesla Inc",
    "TSM": "Taiwan Semiconductor Mfg. Co. Ltd.",
    "UNH": "UnitedHealth Group Inc",
    "V": "Viza Inc",
    "WMT": "Walmart Inc"
}


# create class to load stocks from a CSV file into a data frame in the application
def loadStock(stock_sbl):

    # define the path for the requested stock using the user inputed stock symbol
    csv_path = './stocks/%s.csv' % stock_sbl

    # check if csv_path exists, if exists load the stock, else return an empty dataframe
    if os.path.exists(csv_path):
        stock = pd.read_csv(csv_path, index_col=[0], parse_dates=[0])
    else:
        stock = pd.DataFrame()
        return stock

    # remove the 'Close' column from the data frame and return the updated stock
    del stock['Close']
    stock = stock[['Open', 'High', 'Low', 'Volume', 'Adj Close']]

    return stock


# determine the primary features of the stock data using PCA
def featureAnalyse(stock):

    # load the stock columns into a variable x and scale the data
    x = np.array(stock[['Open', 'High', 'Low', 'Volume', 'Adj Close']])
    x = preprocessing.scale(x)

    # remove rows that are missing data
    stock.dropna(inplace=True)

    # define a variable for PCA and fit the x values
    pca = PCA()
    pca.fit(x)

    per_var = np.round(pca.explained_variance_ratio_, decimals=1)

    # create and display a graph showing the PCA results
    features = ['Open', 'High', 'Low', 'Volume', 'Adj Close']
    plt.figure(figsize=(10, 5))
    plt.bar(x=range(1, len(per_var)+1), height=per_var, tick_label=features)
    plt.xlabel('Features')
    plt.ylabel('Explained Variance Ration')
    plt.show()


# analyze passed in stock to determine future stock values
def predictStock(input_stock):

    # remove rows with missing values
    input_stock.dropna(inplace=True)

    # assign the open and high columns to the variable x and scale the data
    x = np.array(input_stock[['Open', 'High']])
    x = preprocessing.scale(x)
    x_predict = x[-prediction_length:]

    # assign the adj close column to the variable y
    y = np.array(input_stock['Adj Close'])

    # split the data into test and train variables
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    # define linear regression and fit the model with the training data
    lin_reg = LinearRegression()
    lin_reg.fit(x_train, y_train)

    # determine the accuracy of the prediction model
    global prediction_accuracy
    y_predicted = lin_reg.predict(x_test)
    avg_mean = input_stock['Adj Close'].mean()
    error_mean = np.sqrt(mse(y_test, y_predicted))
    prediction_accuracy = (((avg_mean - error_mean)/avg_mean) * 100)

    # predict future stocks using the linear regression model
    predicted_stocks = lin_reg.predict(x_predict)

    return predicted_stocks


# graph the stock data, and future stock predictions
def plotStocks(predicted_stock, stock, stock_data, forecast_data, error_data):

    # add a new column named Forecast and set values to nan
    stock['Forecast'] = np.nan

    # set stock_date to the last index in the stock data frame which is a date
    stock_date = stock.iloc[-1].name

    # set the last forecast value to the same as the last adj close value
    stock.loc[stock_date, 'Forecast'] = stock.loc[stock_date, 'Adj Close']

    # loop through each value of predicted_stock and add the value to the forecast column for the next day
    for i in predicted_stock:
        stock_date += datetime.timedelta(days=1)
        stock.loc[stock_date, 'Forecast'] = i

    # if the passed in stock_data variable is true, graph the historic stock data
    if stock_data:
        stock['Adj Close'].plot(color='black')

    # if the passed in forecast_data variable is true, graph the forecasted data
    if forecast_data:
        stock['Forecast'].plot(color='red')
        print('The prediction accuracy is: %.2f%%\n' % prediction_accuracy)

    # if the passed in error_data variable is true, shade above and below the forecasted data
    if error_data:
        plt.fill_between(stock.index, stock['Forecast'] - (100 - prediction_accuracy), stock['Forecast'] +
                         (100 - prediction_accuracy), alpha=1, edgecolor='gray', facecolor='lightgray')

    # label and show plotted graphs
    plt.title('Closing Price vs. Date ')
    plt.subplots_adjust(0.05, 0.10, 0.95, 0.95)
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.show()


# allow the user to change the prediction length
def predictionLength():

    status = False

    # while status is false loop until user gives a number between 1 and 100
    while not status:
        global prediction_length
        print('Current prediction length: %d days' % prediction_length)

        p_days = input('New prediction length (Between 1 - 100):')

        # check to ensure the entered value is a decimal
        if p_days.isdecimal():
            # check that the entered decimal is greater than 0 and less than or equal to 100
            if 0 < int(p_days) <= 100:
                print('Prediction days updated to %s ' % p_days)
                prediction_length = int(p_days)
                status = True
            else:
                print('Number of days must be between 1 and 100')
        else:
            print('Invalid number of days')


# display all of the offered stocks
def viewAvbStock():

    # loop through the avb_stock dictionary and print the values
    for i in avb_stock:
        print(i, " - ", avb_stock[i])


choice = ''

# loop through the user dashboard until the user types Q to quit

while choice != 'Q':

    print('______________________________________________________________________')
    print('')
    print('Enter the desired stock symbol to analyze the stock or choose an option from below:')
    print('')
    print('A = View available stock symbols')
    print('Q = Quit')
    print('----------------------------------------------------------------------')
    choice = input("Choice: ").upper()

    if choice == 'Q':
        break
    elif choice == 'A':

        viewAvbStock()
    else:
        if loadStock(choice).empty:
            print('Stock not found')

        else:
            selected_stock_sbl = choice
            while choice != 'Q':
                print('______________________________________________________________________')
                print(avb_stock[selected_stock_sbl.upper()])
                print('Choose an option:')
                print('1 = View Feature Analyses Graph')
                print('2 = View Stock Prediction Graph')
                print('3 = View Only Historic Stock Data')
                print('4 = View Only Forecast Stock Data')
                print('5 = View Prediction Graph With Error Shaded')
                print('0 = Choose A New Stock')
                print('P = Change The Number Of Days To Predict (Current = %d days)' % prediction_length)
                choice = input().upper()

                selected_stock = loadStock(selected_stock_sbl)

                if choice == '1':
                    featureAnalyse(selected_stock)
                elif choice == '2':
                    plotStocks(predictStock(selected_stock), selected_stock, True, True, False)
                elif choice == '3':
                    plotStocks(predictStock(selected_stock), selected_stock, True, False, False)
                elif choice == '4':
                    plotStocks(predictStock(selected_stock), selected_stock, False, True, False)
                elif choice == '5':
                    plotStocks(predictStock(selected_stock), selected_stock, False, True, True)
                elif choice == '0':
                    break
                elif choice == 'P':
                    predictionLength()









