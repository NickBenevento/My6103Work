# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dm6103 as dm
plt.style.use('classic')

# The dataset is obtained from 
# https://gssdataexplorer.norc.org 
# for you here. But if you are interested, you can try get it yourself. 
# create an account
# create a project
# select these eight variables: 
# ballot, id, year, hrs1 (hours worked last week), marital, 
# childs, income, happy, 
# (use the search function to find them if needed.)
# add the variables to cart 
# extract data 
# name your extract
# add all the 8 variables to the extract
# Choose output option, select only years 2000 - 2018 
# file format Excel Workbook (data + metadata)
# create extract
# It will take some time to process. 
# When it is ready, click on the download button. 
# you will get a .tar file
# if your system cannot unzip it, google it. (Windows can use 7zip utility. Mac should have it (tar function) built-in.)
# Open in excel (or other comparable software), then save it as csv
# So now you have Happy table to work with
#
# When we import using pandas, we need to do pre-processing like what we did in class
# So clean up the columns. You can use some of the functions we defined in class, like the total family income, and number of children. 
# Other ones like worked hour last week, etc, you'll need a new function. 
# Happy: change it to numeric codes (ordinal variable)
# Ballot: just call it a, b, or c 
# Marital status, it's up to you whether you want to rename the values. 
# 
#
# After the preprocessing, make these plots
# Box plot for hours worked last week, for the different marital status. (So x is marital status, and y is hours worked.) 
# Violin plot for income vs happiness, choose one sensible variable to color (hue) the plot. 
# Use happiness as numeric, make scatterplot with jittering in both x and y between happiness and number of children. Choose what variable you want for hue/color.
# If you have somewhat of a belief that happiness is caused/determined/affected by number of children, or the other 
# way around (having babies/children are caused/determined/affected by happiness), then put the dependent 
# variable in y, and briefly explain your choice.

dfhappy = dm.api_dsLand('Happy') 

# ----------- PREPROCESSING --------------
# %%
# HAPPY
dfhappy['happy'] = dfhappy.happy.map(lambda x: np.nan if x.strip()=='Don\'t know' 
                                        else np.nan if x.strip()=='Not applicable'
                                        else np.nan if x.strip()=='No answer'
                                        else 0 if x.strip()=='Not too happy'
                                        else 1 if x.strip()=='Pretty happy'
                                        else 2 if x.strip()=='Very happy' else x)
# dfhappy.head()
# print(dfhappy.happy.unique())

# %%
# BALLOT
dfhappy['ballet'] = dfhappy['ballet'].str.replace('Ballot ', '')

# %%
# HOURS
print(dfhappy.hrs1.unique())
dfhappy['hrs1'] = pd.to_numeric(dfhappy['hrs1'], errors='coerce')
print(dfhappy.hrs1.unique())

# %%
# CHILDREN
dfhappy['childs'] = dfhappy.childs.map(lambda x: np.nan if x.strip()=='Dk na' 
                                        else 8 if x.strip()=='Eight or m'
                                        else int(x))

# %%
def cleanDfIncome(row, colname): # colname can be 'rincome', 'income' etc
  thisamt = row[colname].strip().lower()
  if (thisamt == "don't know"): return np.nan
  if (thisamt == "no answer"): return np.nan
  if (thisamt == "refused"): return np.nan 
  if (thisamt == "Lt $1000"): return np.random.uniform(0,999)
  if (thisamt == "$1000 to 2999"): return np.random.uniform(1000,2999)
  if (thisamt == "$3000 to 3999"): return np.random.uniform(3000,3999)
  if (thisamt == "$4000 to 4999"): return np.random.uniform(4000,4999)
  if (thisamt == "$5000 to 5999"): return np.random.uniform(5000,5999)
  if (thisamt == "$6000 to 6999"): return np.random.uniform(6000,6999)
  if (thisamt == "$7000 to 7999"): return np.random.uniform(7000,7999)
  if (thisamt == "$8000 to 9999"): return np.random.uniform(8000,9999)
  if (thisamt == "$10000 - 14999"): return np.random.uniform(10000,14999)
  if (thisamt == "$15000 - 19999"): return np.random.uniform(15000,19999)
  if (thisamt == "$20000 - 24999"): return np.random.uniform(20000,24999)
  if (thisamt == "$25000 or more"): return ( 25000 + 10000*np.random.chisquare(2) )
  return np.nan
# end function cleanDfIncome
print("\nReady to continue.")

# Now apply to df row-wise. 
# Here with two arguments in the function, we use this syntax
print(dfhappy.head())
print(dfhappy.income.unique())
dfhappy['income'] = dfhappy.apply(cleanDfIncome, colname='income', axis=1)
print(dfhappy.head())


# ----------- PLOTS --------------
# %%
# HOURS WORKED vs. MARITAL, BOX PLOT

# get each of the different labels (drop 'No answer')
labels = dfhappy.marital.unique()[:-1]
# create a new df for each marriage status
statuses = []
for label in labels:
    df = dfhappy[dfhappy['marital'] == label].dropna(axis=0, how='any')
    statuses.append(df)


fig, ax = plt.subplots()
# display hours worked for each marriage status on same plot
data = [ status['hrs1'] for status in statuses]
plt.boxplot(data, labels=labels)
plt.xlabel('Marital Status')
plt.ylabel('Hours Worked (weekly)')

plt.show()

# %%
# INCOME vs. HAPPINESS, VIOLIN PLOT
values = dfhappy.happy.unique()[:-1]
i0 = dfhappy[dfhappy['happy'] == 0].dropna(axis=0, how='any')
i1 = dfhappy[dfhappy['happy'] == 1].dropna(axis=0, how='any')
i2 = dfhappy[dfhappy['happy'] == 2].dropna(axis=0, how='any')


fig, ax = plt.subplots()
# display income for each happiness on same plot
data = [ i0['income'], i1['income'], i2['income']]
plt.violinplot(data)
plt.xlabel('Happiness')
plt.ylabel('Income')

plt.show()
# %%
# HAPPINESS vs. NUMBER OF CHILDREN, SCATTER PLOT
fuzzy_happy = dfhappy.happy + np.random.normal(0,1, size=len(dfhappy.happy))
fuzzy_children = dfhappy.childs + np.random.normal(0,1, size=len(dfhappy.childs))

print(fuzzy_happy)
# plt.plot(fuzzy_children, fuzzy_happy, 'o', c=fuzzy_happy[1])
# plt.plot(fuzzy_children, fuzzy_happy, 'o', c=fuzzy_happy[1], cmap='gray')
plt.scatter(fuzzy_children, fuzzy_happy, s=100, c=fuzzy_happy, cmap='Blues')
plt.xlabel('Number of children')
plt.ylabel('Happiness')

# I put the happiness value as the dependent variable, as I thought there might
# interesting happiness results based on the number of children. Is there a 
# fluctuation, where a certain number of children brings more happiness
# than other amounts of children? Is there a number of children that is 
# "too much"?


# %%
