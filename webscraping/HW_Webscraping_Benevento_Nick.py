# -*- coding: utf-8 -*-
# The line above is used when I need to run this file as a shell script, as a cron job

#%%[markdown]
#
# # HW Web Scraping
# ## Calling on all Weather enthusiasts

# Reference : class file WS_sample_Stock.py
#
# Okay, I want to collect weather info for my family living at different zip codes. 
# Feel free to replace the zipcode list with your favorites. 
# 
# Ideally, I can put the codes on pythonanywhere.com,  
# so that it can be set as a cron job on a reliable server 
# instead of on my own laptop.  
# As we did in class, we can use weather.gov to get the weather info 
# for the different zip codes. But the way that website is designed, 
# we cannot encode the zip code into the url. So we used 
# selenium to automate. But we cannot install selenium on 
# pythonanywhere.com. That is using too much resources on the public server.
# 
# Good thing there are so many options out there. I found 
# [this site](url = 'https://www.wunderground.com/weather/us/dc/20052') works okay.
# 
# Now, get the codes to work, with a list of 10 zip codes with the state abbreviation 
# ( so the list/tuple should looks like this: ('dc/20052' , 'ny/10001' , 'ca/90210' , ... ) ), 
# automatically pull the forecast temperature high and low for the day 
# at 6am at our Washington-DC time? 
# The stock portfolio example (WS_sample_Stock.py) in class is adaptable to 
# handle this task. 
# 
# Have the codes tested and working on your computer. If you are interested, you can 
# deploy it pythonanhywhere.com. I can show you how. Deploying to pythonanywhere.com is 
# an optional exercise. Simply run your 
# codes on your computer for a few days, or 2-3 times a day to get 
# the temp info at different times to make sure it works. 
# # The csv file will eventually looks like this
# Date,dc/20052,ny/10001,ca/90210,nj/07069,va/22207,il/60007,tx/77001,az/85001,pa/19019,tx/78006
# 2022-03-03,53° | 34°,50° | 34°,--° | 53°,51° | 29°,53° | 34°,30° | 22°,74° | 56°,71° | 49°,53° | 32°,71° | 51°
# 2022-03-02,53° | 52°,73° | 39°,68° | --°,62° | 32°,48° | 31°,40° | 25°,69° | 52°,72° | 52°,65° | 37°,66° | 56°
# 2022-03-01,53° | 35°,49° | 34°,--° | 53°,51° | 29°,52° | 34°,30° | 21°,74° | 55°,--° | 47°,53° | 32°,71° | 51°
# 
# Some of the temperatures might come up as --°, which is fine. Don't worry about that.
# That's all you need to submit, the working codes and a sample data csv file. 
#  
# Imagine you can do something similar to track your favorite sport team's statistics, or air ticket price, ...
# What get you excited?
# 
# Of course, whenever there is an API available, use that instead of webscraping. It's much more reliable.

#%%

from re import L
from matplotlib.pyplot import axis
import numpy as np
import requests
import datetime
import pandas as pd
import requests
from bs4 import BeautifulSoup

def getUrl(zipcode):
  url = 'https://www.wunderground.com/weather/us/'+zipcode # zipcode should be in the format:  stateAbbreviation/five-digit-zipcode like dc/20052
  return url 

def getSoup(url,parser=''):
  # ######  QUESTION 1      QUESTION 1      QUESTION 1   ##########
  # write your codes here
  # p = parser if ( parser=='lxml' or parser=='html.parser') else 'html5lib'
  p = parser if ( parser=='lxml' or parser=='html.parser') else 'html.parser'
  r = requests.get(url)
  s = BeautifulSoup(r.content, p) # or 'lxml' or 'html.parser'

  # ######  END of QUESTION 1    ###   END of QUESTION 1   ##########
  return s

def getTempHiLo(soup): # get the block of values of hi-lo temperature on this site
  # ######  QUESTION 2      QUESTION 2      QUESTION 2   ##########

  # write your codes here
  
  high = soup.select_one('div.hi-lo > span.hi').text
  low = soup.select_one('div.hi-lo > span.lo').text

  # ######  END of QUESTION 2    ###   END of QUESTION 2   ##########
  return high, low

def getDailyTemp(filename):
  # the source file header has the list of zip codes I want to keep track. 
  # I am using this source file to keep track of the list of zipcodes below: 
  # zipcodes = ['dc/20052' , 'ny/10001' , 'ca/90210', 'nj/07069', 'va/22207', 'il/60007', 'tx/77001', 'az/85001', 'pa/19019', 'tx/78006']
  
  # I will use the date string as the key for my dataframe
  tday = datetime.datetime.today()
  tdaystr = tday.strftime('%Y-%m-%d')  # default format is 'yyyy-mm-dd 00:00:00'

  # open file, import df from file, with first row as header, first column as index.
  df_last = pd.read_csv(filename, header=0, index_col=0, date_parser=lambda x: datetime.datetime.strptime(x, '%Y-%m-%d') )

  df_new = pd.DataFrame(columns=df_last.columns ) # set a new empty dataframe for new day's data
  df_new.index.name = df_last.index.name # set the index name 
  df_new = df_new.append(pd.Series(name=tdaystr)) # add a blank row with today's date as index

  # ######  QUESTION 3      QUESTION 3      QUESTION 3   ##########

  # write your codes here 
  # You can run the current codes to see what is the df_new and df_last look like. 
  # Need to get the new Temperatures for each location/zip, and put them in the new data frame 
  # Then insert that df_new to the top of df_last.
  zipcodes = ['dc/20052' , 'ny/10001' , 'ca/90210', 'nj/07069', 'va/22207', 'il/60007', 'tx/77001', 'az/85001', 'pa/19019', 'tx/78006']
  for code in zipcodes:
    url = getUrl(code)
    soup = getSoup(url)
    high, low = getTempHiLo(soup)
    df_new.loc[:, code] = low + ' | ' + high

  # df_new = df_new.append(df_last)
  df_last = pd.concat([df_new, df_last])
  # ######  END of QUESTION 3    ###   END of QUESTION 3   ##########
  
  df_last.index = pd.to_datetime(df_last.index, format='%Y-%m-%d') # row index = 'Date', fixing formatting issue. Without this line, the dates on dataframe will be recorded as 'yyyy-mm-dd 00:00:00' to the datafame

  df_last = df_last.iloc[0:25,:]   # trim number of rows to max 25, before saving to file to prevent file growing too big
  df_last.to_csv(filename, encoding='utf_8_sig')  # saving/updating csv file    # df_last.to_csv(filename, sep='\t')

  return None



#%%
# make sure the folder is correct
source = 'weatherDaily.csv'
# run one time to test getting weather data for today
getDailyTemp(source)
# %%
