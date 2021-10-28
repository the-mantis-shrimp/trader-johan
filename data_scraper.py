import requests
import csv
import os
import pandas as pd
import glob
from datetime import datetime
import time
from array import array
import sys
#from os import path

def ticker_list():
    with open('ticker_data.csv', newline='') as ticker_file:
        reader = csv.DictReader(ticker_file)
        tickers = [row['Symbol'] for row in reader]
        for line in tickers:
            tickers[tickers.index(line)] = line.replace('/', '-')
    return tickers

def scrape(ticker):
    url = f"https://api.tdameritrade.com/v1/marketdata/{ticker}/pricehistory?apikey=ERG75M6AUJLRG4SUVBGCMRRG4JIR3VCZ&periodType=year&period=10&frequencyType=daily"

    req = requests.get(url)

    data = req.content

    json_file = open(f"./jsons/{ticker}.json", 'wb')
    json_file.write(data)
    json_file.close()

def json_to_csv(tickers):
    for ticker in tickers:
        print(f"Converting {ticker} to csv...")
        try:    
            x = pd.read_json(f"./jsons/{ticker}.json")
            x.to_csv(f"./csvs/{ticker}.csv")
        except:
            print("ERROR: Moving on...")

def format_csv(tickers):
    fields = ['ticker', 'open', 'high', 'low', 'close', 'volume', 'datetime', 'empty']
    for line in tickers:
        if path.exists(f"./csvs/{line}.csv"):
            with open(f"./csvs/{line}.csv", newline='') as x:
                reader = csv.reader(x)
                rows = list(reader)
                rows[0] = fields
                for i in rows[1:]:  
                    # remove a bunch of crap
                    i[1] = i[1].replace('{', "")
                    i[1] = i[1].replace('}', "")
                    i[1] = i[1].replace(" '", "'")
                    i[1] = i[1].replace("'open': ", "")
                    i[1] = i[1].replace("'high': ", "")
                    i[1] = i[1].replace("'low': ", "")
                    i[1] = i[1].replace("'close': ", "")
                    i[1] = i[1].replace("'volume': ", "")
                    i[1] = i[1].replace("'datetime': ", "")
                    final_data = i[1].split(',')
                    
                    # Convert time_stamp to readable format
                    time_stamp = int(final_data[5]) 
                    time_stamp = time_stamp/1000
                    time_stamp = datetime.utcfromtimestamp(time_stamp).strftime('%Y-%m-%d')
                    final_data[5] = time_stamp

                    tf = i[3] # preserve T/F
                    i.pop(0)  # remove old row number
                    i.pop(0)  # remove unformatted data
                    i.pop(1)  # remove T/F
                    i.extend(final_data)   # add formatted data
                    i.append(tf)  # add T/F
                    
            print(f"Reformatting {line}.csv") 
            with open(f"./csvs/{line}.csv", 'w', newline='') as x:
                writer = csv.writer(x)
                writer.writerows(rows)

def combine_csv():
    os.chdir("./csvs")
    extension = 'csv'
    all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
    #combine all files in the list
    combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
    combined_csv = combined_csv.dropna(how='all')
    #export to csv
    combined_csv.to_csv( "combined_csv.csv", index=False, encoding='utf-8-sig')
    os.chdir("..")


######## Main Code ########
tickers = ticker_list()

# iterate through ticker calls from ameritrade API
for i in tickers:
    try:
        print(f"scraping {tickers.index(i)}: {i}")
        csv_file = scrape(i)
        if not tickers.index(i) % 80:
            print(f"{round(tickers.index(i) / 7980 * 100, 1)}% complete. Waiting for refresh...")
            time.sleep(75)
    except:
        print(f"ERROR: Couldn't scrape {i}!")

#tickers = ['A']   # can be used to test a single ticker

# Note: Must run both functions to recieve a properly formatted csv
#json_to_csv(tickers)
#format_csv(tickers)

# function to --guess what-- combine csv files into one massive file.
# combine_csv()