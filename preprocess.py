import numpy as np
from PIL import Image

import os
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import dateutil
from matplotlib.pyplot import savefig



def plot_selected(df, columns,symbol, start_index, end_index):
    """Plot the desired columns over index values in the given range."""
    
    plot_data(df.ix["{}".format(start_index):"{}".format(end_index),["{}".format(symbol)]],title='Selected data')#caution this independent expression for circulation


def symbol_to_path(symbol, base_dir="E://data"):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))


def get_data(symbols, dates):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)
    if 'SPY' not in symbols:  # add SPY for reference, if absent
        symbols.insert(0, 'SPY')

    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df = df.join(df_temp)
        if symbol == 'SPY':  # drop dates SPY did not trade
            df = df.dropna(subset=["SPY"])

    return df

def normalize_data(df):
    return df/df.ix[0,:]

def plot_data(df, title="Stock prices"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    df = normalize_data(df)
    ax = df.plot(title=title, fontsize=12)
    plt.axis('off')
    plt.title(' ')
    plt.legend('off')
    # ax.set_xlabel("Date")
   # ax.set_ylabel("Price")
    

def picturetomatrix(filename):
    im = Image.open('e:/picture/{}.jpg'.format(filename))
    im = im.convert("L")
    data = im.getdata()
    data = np.matrix(data)
    print(data)
    data = np.reshape(data,(288,-1))
    new_im = Image.fromarray(data)
    np.savetxt('e:/picture/{}.csv'.format(filename),data,delimiter = ',')
    #new_im.show()
    

def test_run():
    # Define a date range
    dates = pd.date_range('2016-03-01', '2017-04-03')

    # Choose stock symbols to read
   # symbols = ['AAPL', 'FB', 'GE','AMZN','JNJ','JPM','WFC','XOM']  # SPY will be added in get_data()
    #symbols = ['1','2',	'3',	'4',	'5',	'6',	'7',	'8',	'9']
    symbols = ['1','2']
    # Get stock data
    df = get_data(symbols, dates)
    begin = datetime.date(2016,6,1)  
    end = datetime.date(2016,6,5)
    
    
    # Slice and plot
    for symbol in symbols:
        print(symbol)
        a = begin
        b = begin + dateutil.relativedelta.relativedelta(days=31)
        for i in range((end-begin).days):
             day = begin + datetime.timedelta(days=i)  
             a = day
             b = day + datetime.timedelta(days = 30)
             print("i:"+str(i))
             print(a)
             print("c"+str(b))
             plot_selected(df, ['SPY', 'FB'],symbol, a, b)
             filename = str(symbol)+"_"+str(i)+"_"+str(a)
             plt.savefig('E:/picture/{}.jpg'.format(filename))
             picturetomatrix(filename)
             plt.show()
             print("________")
        #plot_selected(df, ['SPY', 'FB'],symbol, '2016-04-01', '2017-04-01')#Seems to be no use in changing spy here
   # plot_data(df, title="Stock prices")
   # print(df)

if __name__ == "__main__":
  
    test_run()
   




