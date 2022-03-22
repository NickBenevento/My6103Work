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
# The general principles I will be looking at to determine whether or not the world is a utopia
# mainly deal with equality across genders/ethnicities, distribution of education and income,
# and otherwise happiness, mainly denoted through divorce rates. I do believe in capitolism, as I 
# feel that innovation and hard work should be rewarded. However, in a perfect world there isn't an uber
# rich upper class that overshadows a lot of people living in poverty. So, I will also try to look at the
# distribution of incomes to see if a disproportionate amount of people are living in the lower class.

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
plt.savefig('world1_gender_income')
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
plt.title('world1_ethnicity_income')

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
# INCOME DISTRIBUTION:

income_brackets = [0, 0, 0, 0, 0, 0, 0, 0, 0]

for income in world1.income00:
    if income < 20000:
        income_brackets[0] += 1
    elif income < 40000:
        income_brackets[1] += 1
    elif income < 60000:
        income_brackets[2] += 1
    elif income < 80000:
        income_brackets[3] += 1
    elif income < 100000:
        income_brackets[4] += 1
    elif income < 120000:
        income_brackets[5] += 1
    elif income < 140000:
        income_brackets[6] += 1
    elif income < 160000:
        income_brackets[7] += 1
    else:
        income_brackets[8] += 1

labels = ['< 20,000', '20,000-40,000', '40,000-60,000', '60,000-80,000', '80,000-100,000', 
        '100,000-120,000', '120,000-140,000', '140,000-160,000', '> 160,000']
print(income_brackets)
plt.bar(labels, income_brackets, width=0.4)
plt.xticks(rotation=90)
plt.xlabel('Yearly income ($)')
plt.ylabel('People')
plt.title('Income Brackets')
plt.savefig('world1_Income_Distribution')
print('Highest income: ', world1['income00'].max())
print('Number of people making < $20,000: ', income_brackets[0])
# As we can see from the graph, there is what seems to be a relatively fair
# distribution of incomes across the population. There are people that are in the upper
# class making more money, but the total income isn't disproportinate to the rest
# of the population. In fact, the person with the highest income is only making
# $162,668, which is not leagues beyond what the rest of the population is earning.
# Additionally, there are only 2 people making less than $20,000.

#%%
# EDUCATION
plot = world1.loc[:, 'education'].value_counts().sort_index().plot(kind='bar')
plot.get_figure().savefig('world1_education.png')
plt.title('Education Distribution')
plt.xlabel('Years of Education')
plt.ylabel('Number of People')
plt.savefig('world1_Education')

average_education = world1.loc[:, 'education'].mean()
mode_education = world1.mode()['education'][0]
print('Average education level: ', average_education)
print('Most common education level: ', mode_education)

# Looking at the years of education across the population, we can see that the majority of the people
# have over 11 years of schooling. The average education level is about 15 years, with most people having 16
# years of education. This is a very educated population, which is in my opinion a great benefit to a society.
# An educated population can better understand things like policy changes, scientific reasoning, etc.
# Generally speaking, it also allows for better innovation and advancement of technology.


# As a final conclusion, I would not say that this world is a utopia, but it
# does have some desirable qualities in terms of education and income distribution.
# On the downside, there seems to be some descrimination on income based on gender
# and ethnicity, which is definitiely not okay.

# ------------ END WORLD 1 --------------------



# ------------ WORLD 2 --------------------


# GENDER INCOME
men = world2[world2.gender == 1]
m_avg = men.loc[:, 'income00'].mean()
m_med = men.loc[:, 'income00'].median()
women = world2[world2.gender == 0]
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
plt.savefig('world2_gender_income')
print(m_avg)
print(w_avg)
print(m_avg - w_avg)
print(m_med - w_med)

# In the second world, we can see that there is almost no difference in the average
# and median income values for men and women. In comparison to world1, this is 
# a good sign that equality is valued.


#%%
# ETHNICITY INCOME
world2.head()
ethnicity1 = world2[world2.ethnic == 0]
e1_avg = ethnicity1.loc[:, 'income00'].mean()
e1_med = ethnicity1.loc[:, 'income00'].median()

ethnicity2 = world2[world2.ethnic == 1]
e2_avg = ethnicity2.loc[:, 'income00'].mean()
e2_med = ethnicity2.loc[:, 'income00'].median()

ethnicity3 = world2[world2.ethnic == 2]
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
plt.title('world2_ethnicity_income')

plt.savefig('world2_ethnicity_income')


print(e3_med - e1_med)
print(e3_med - e2_med)

# Looking at the mean and median income across ethnicities, we can see that there
# is virtually no bias in terms of ethnicities with relation to income. Each ethnicity
# has nearly identical average and median income values. As stated in the previous
# section, this is another strong indicator that equality is valued, which is
# a key characteristic of a utopia.

#%%
# DIVORCE RATES
all_marriages = world2[world2.marital.isin([1, 2, 3])]
print('all marriages: ', len(all_marriages))
divorce = len(all_marriages[all_marriages.marital == 2])
print('divorce: ', divorce)
print('divorce rate: ', divorce / len(all_marriages))


# The divorce rates are very similar to those of world 1; of the people in world 2
# who were married at one point, about 14.2% of them got divorced. Again, this
# is a very reasonable, fairly low number that could indicate general happiness
# among the population.
#%%
# INCOME DISTRIBUTION

income_brackets = [0, 0, 0, 0, 0, 0, 0, 0, 0]

for income in world2.income00:
    if income < 20000:
        income_brackets[0] += 1
    elif income < 40000:
        income_brackets[1] += 1
    elif income < 60000:
        income_brackets[2] += 1
    elif income < 80000:
        income_brackets[3] += 1
    elif income < 100000:
        income_brackets[4] += 1
    elif income < 120000:
        income_brackets[5] += 1
    elif income < 140000:
        income_brackets[6] += 1
    elif income < 160000:
        income_brackets[7] += 1
    else:
        income_brackets[8] += 1

labels = ['< 20,000', '20,000-40,000', '40,000-60,000', '60,000-80,000', '80,000-100,000', 
        '100,000-120,000', '120,000-140,000', '140,000-160,000', '> 160,000']
print(income_brackets)
plt.bar(labels, income_brackets, width=0.4)
plt.xticks(rotation=90)
plt.xlabel('Yearly income ($)')
plt.ylabel('People')
plt.title('Income Brackets')
plt.savefig('world2_Income_Distribution')
print('Highest income: ', world2['income00'].max())
print('Number of people making < $20,000: ', income_brackets[0])

# As we can see from the graph, there is what seems to be a relatively fair
# distribution of incomes across the population. There are people that are in the upper
# class making more money, but the total income isn't disproportinate to the rest
# of the population. Similar to world 1, the person with the highest income is only making
# $161,737, which is not leagues beyond what the rest of the population is earning.
# There are 5 people making under $20,000 / year, but the majority of the population
# is earning between $40,000 and $60,000 a year.

#%%
# EDUCATION

plot = world2.loc[:, 'education'].value_counts().sort_index().plot(kind='bar')
plot.get_figure().savefig('world2_education.png')
plt.title('Education Distribution')
plt.xlabel('Years of Education')
plt.ylabel('Number of People')
plt.savefig('world2_Education')

average_education = world2.loc[:, 'education'].mean()
mode_education = world2.mode()['education'][0]
print('Average education level: ', average_education)
print('Most common education level: ', mode_education)

# Looking at the years of education across the population, we can see that the majority of the people
# have over 11 years of schooling. The average education level is just over 15 years, with most people having 16
# years of education. Again, this is very similar to the world 1 in terms of 
# level of education. This is another sign that this population could be a utopia.


# As a final conclusion, I would say that world 2 is in fact a utopia. It has the
# the same desirable qualities that world 1 has in terms of education levels, divorce
# rates, and income distribution. It also does not seem to have any sort of discrimination
# in terms of how gender and ethnicity affect income, which means that equality
# is a core component of this world. Combining these factors together, world 2
# seems to be a very fair and pleasant world for its citizens to live in.
# ------------ END WORLD 2 --------------------
