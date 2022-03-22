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

# The general principles I will be looking at to determine whether or not the world is a utopia
# mainly deal with equality across genders/ethnicities, distribution of education and income,
# and otherwise happiness, mainly denoted through divorce rates. I do believe in capitolism, as I 
# feel that innovation and hard work should be rewarded. However, in a perfect world there isn't an uber
# rich upper class that overshadows a lot of people living in poverty. So, I will also try to look at the
# distribution of incomes to see if a disproportionate amount of people are living in the lower class.

# income distribution vs ethnicity --> is 1 group making more than the others?
# same as above but for men vs women
# divorce rates
# distribution of income --> a lot in poverty? is top 1% making way more?

# ------------ WORLD 1 --------------------
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
fig, ax = plt.subplots()
p1 = plt.bar(ind, (m_avg, m_med), width, label='Men')
p2 = plt.bar(ind + width, (w_avg, w_med), width, label='Women')
plt.xticks(ind + width / 2, ('Average Income', 'Median Income'))
ax.bar_label(p1)
ax.bar_label(p2)
plt.margins(y=0.25)
plt.ylabel('Income ($)')
plt.title('Men vs. Women Income')
plt.legend(loc='upper center')
plt.savefig('world1_men_women_income')
print(m_avg)
print(w_avg)
print(m_avg - w_avg)
print(m_med - w_med)

# In the first world, we can see that men make significantly more on average than women do: about $10,818.
# The median income for men is also $12,430 more than the median income for women. This is an indication
# of inequality between the pay for men and women, which is not something that is desired in a utopia.


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


print(e3_med - e1_med)
print(e3_med - e2_med)

# Looking at the mean and median income across ethnicities, we can see that ethnicity 1 and ethnicity 2 are
# fairly similar, with a difference of only $3,820 in the mean income and $3,025 in the median income.
# However, residents that are of ethnicity 3 make significantly more than those of ethnicity 1 or 2. 
# We can see that those of ethnicity 3 make $15,221 more on average than those of ethnicity 1, and $19,0424
# more than those of ethnicity 2. In terms of median income, they make $17,135 more than ethnicity 1 people
# and $20,160 more than ethnicity 2 people. These are alarming differences, as they seem to indicate that
# people of one ethnicity have much higher incomes than everyone else. While this could be attributed to other
# external factors, going based on the available data, it is a red flag that not everyone in this population is
# treated equally.

#%%
# DIVORCE RATES
all_marriages = world1[world1.marital.isin([1, 2, 3])]
print('all marriages: ', len(all_marriages))
divorce = len(all_marriages[all_marriages.marital == 2])
print('divorce: ', divorce)
print('divorce rate: ', divorce / len(all_marriages))


# While divorce rates are not a perfect way of determining how happy a population is, a lower divorce rate 
# generally indicates that the couples in the relationship are more happy and more likely to stay together.
# Since people may get divorced for a variety of reasons, divorce is not always a bad thing, but helps give some
# insight into the population. Of the people in world 1 who were married at one point, about 14.6% of them
# got divorced. Comparing this to the fact that almost 50% of all marriages in the United States end in 
# divorce or separation, this is a very reasonable figure that could indicate people are generally 
# happier in this population.
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
plot = world1.loc[:, 'education'].value_counts().sort_index().plot(kind='bar')
plot.get_figure().savefig('world1_education.png')
plt.title('Education Distribution')
plt.xlabel('Years of Education')
plt.ylabel('Number of People')

average_education = world1.loc[:, 'education'].mean()
mode_education = world1.mode()['education'][0]
print('Average education: ', average_education)
print('Most common education: ', mode_education)

# Looking at the years of education across the population, we can see that the majority of the people
# have over 11 years of schooling. The average education level is about 15 years, with most people having 16
# years of education. This is a very educated population, which is in my opinion a great benefit to a society.
# An educated population can better understand things like policy changes, scientific reasoning, etc.
# Generally speaking, it also allows for better innovation and advancement of technology.

# ------------ END WORLD 1 --------------------



# ------------ WORLD 2 --------------------

# %%

# ------------ END WORLD 2 --------------------