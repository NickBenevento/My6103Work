# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import dm6103 as dm

world1 = dm.api_dsLand('World1', 'id')
world2 = dm.api_dsLand('World2', 'id')

print("\nReady to continue.")


#%% [markdown]
# # Two Worlds 
# 
# I was searching for utopia, and came to this conclusion: If you want to do it right, do it yourself. 
# So I created two worlds. 
#
# Data dictionary:
# * age00: the age at the time of creation. This is only the population from age 30-60.  
# * education: years of education they have had. Education assumed to have stopped. A static data column.  
# * marital: 0-never married, 1-married, 2-divorced, 3-widowed  
# * gender: 0-female, 1-male (for simplicity)  
# * ethnic: 0, 1, 2 (just made up)  
# * income00: annual income at the time of creation   
# * industry: (ordered with increasing average annual salary, according to govt data.)   
#   0. leisure n hospitality  
#   1. retail   
#   2. Education   
#   3. Health   
#   4. construction   
#   5. manufacturing   
#   6. professional n business   
#   7. finance   
# 
# 
# Please do whatever analysis you need, convince your audience both, one, or none of these 
# worlds is fair, or close to a utopia. 
# Use plots, maybe pivot tables, and statistical tests (optional), whatever you deem appropriate 
# and convincing, to draw your conclusions. 
# 
# There are no must-dos (except plots), should-dos, cannot-dos. The more convenicing your analysis, 
# the higher the grade. It's an art.
#

#%%
# ideas:

# income distribution vs ethnicity --> is 1 group making more than the others?
# same as above but for men vs women
# divorce rates
# distribution of income --> a lot in poverty? is top 1% making way more?

#%%
# GENDER INCOME
men = world1[world1.gender == 1]
m_avg = men.loc[:, 'income00'].mean()
m_med = men.loc[:, 'income00'].median()
women = world1[world1.gender == 0]
w_avg = women.loc[:, 'income00'].mean()
w_med = women.loc[:, 'income00'].median()

width = 0.25
ind = np.arange(2)
fig = plt.figure()
plt.bar(ind, (m_avg, m_med), width, label='Men')
plt.bar(ind + width, (w_avg, w_med), width, label='Women')
plt.xticks(ind + width / 2, ('Average Income', 'Median Income'))
plt.ylabel('Income ($)')
plt.title('Men vs. Women Income')
plt.legend(loc='upper center')
plt.savefig('world1_men_women_income')


#%%
# ETHNICITY INCOME
world1.head()
ethnicity1 = world1[world1.ethnic == 0]
e1_avg = ethnicity1.loc[:, 'income00'].mean()
e1_med = ethnicity1.loc[:, 'income00'].median()

ethnicity2 = world1[world1.ethnic == 1]
e2_avg = ethnicity2.loc[:, 'income00'].mean()
e2_med = ethnicity2.loc[:, 'income00'].median()

ethnicity3 = world1[world1.ethnic == 2]
e3_avg = ethnicity3.loc[:, 'income00'].mean()
e3_med = ethnicity3.loc[:, 'income00'].median()

width = 0.2
ind = np.arange(2)
fig, ax = plt.subplots()
p1 = plt.bar(ind, (e1_avg, e1_med), width, label='Ethnicity 1')
p2 = plt.bar(ind + width, (e2_avg, e2_med), width, label='Ethnicity 2')
p3 = plt.bar(ind + 2*width, (e3_avg, e3_med), width, label='Ethnicity 3')
plt.xticks(ind + width, ('Average Income', 'Median Income'))
ax.bar_label(p1)
ax.bar_label(p2)
ax.bar_label(p3)
plt.margins(y=0.25)
plt.ylabel('Income ($)')
plt.legend(loc='upper center')
plt.title('Income Across Ethnicities')

plt.savefig('world1_ethnicity_income')

#%%
# DIVORCE RATES
# all_marriages = world1[world1.marital >= 1]
all_marriages = world1[world1.marital.isin([1, 2, 3])]
print('all marriages: ', len(all_marriages))
divorce = len(all_marriages[all_marriages.marital == 2])
print('divorce: ', divorce)
print('divorce rate: ', divorce / len(all_marriages))


#%%
# education:

# plot = world1.loc[:, 'income00'].plot()
# plot.get_figure().savefig('world1_income.png')
# plot = world1.loc[:, 'education'].value_counts().sort_index().plot(kind='bar')
# plot = world1.loc[:, 'education'].value_counts().sort_index().plot(kind='bar')
plot = world1.loc[:, 'income00'].plot(kind='hist')
plot.get_figure().savefig('world1_education.png')

#%%
# income:

plot = world1.loc[:, 'income00'].value_counts().sort_index().plot()
plot.get_figure().savefig('world1_income.png')

# print(world2.head())
#%%
# plot = world1.loc[:, 'education'].value_counts().plot(kind='bar')
# plot = world1.loc[:, 'education'].value_counts(ascending=True).plot(kind='bar')
plot = world2.loc[:, 'education'].value_counts().sort_index().plot(kind='bar')
plot.get_figure().savefig('world2_income.png')

# %%
# Education vs income
plot = world1.plot(x='education', y='income00')

# %%
