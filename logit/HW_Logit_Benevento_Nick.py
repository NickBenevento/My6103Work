#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import dm6103 as dm
plt.style.use('classic')

# Part I
titanic = dm.api_dsLand('Titanic', 'id')

#%% [markdown]

# # Part I  
# Titanic dataset - statsmodels
# 
# | Variable | Definition | Key/Notes  |  
# | ---- | ---- | ---- |   
# | survival | Survived or not | 0 = No, 1 = Yes |  
# | pclass | Ticket class | 1 = 1st, 2 = 2nd, 3 = 3rd |  
# | sex | Gender / Sex |  |  
# | age | Age in years |  |  
# | sibsp | # of siblings / spouses on the Titanic |  |  
# | parch | # of parents / children on the Titanic |  |  
# | ticket | Ticket number (for superstitious ones) |  |  
# | fare | Passenger fare |  |  
# | embarked | Port of Embarkation | C: Cherbourg, Q: Queenstown, S: Southampton  |  
# 
#%%
# ## Question 1
# With the Titanic dataset, perform some summary visualizations:  
# 
# ### a. Histogram on age. Maybe a stacked histogram on age with male-female as two series if possible
sns.histplot(data=titanic, x='age', hue='sex', multiple='dodge', shrink=0.8)
plt.title('Female and Male Age Histogram')

# %%

# ### b. proportion summary of male-female, survived-dead
titanic.head()
sns.histplot(data=titanic, x='survived', hue='sex', multiple='dodge')

# %%
# ### c. pie chart for “Ticketclass”  
titanic['pclass'].unique()
labels = ['1st', '2nd', '3rd']
values = titanic['pclass'].value_counts().sort_index()
plt.pie(values, labels=labels)
plt.title('Pie chart for Ticketclass')

# %%
# ### d. A single visualization chart that shows info of survival, age, pclass, and sex.
# titanic.head()
sns.set(style="white", palette="muted", color_codes=True)
sns.pairplot(titanic[['survived', 'age', 'pclass', 'sex']], hue='survived')


# %%
# ## Question 2  
# Build a logistic regression model for survival using the statsmodels library. As we did before, include the features that you find plausible. Make sure categorical variables are use properly. If the coefficient(s) turns out insignificant, drop it and re-build.  
import statsmodels.api as sm
from statsmodels.formula.api import glm

model_predictions = pd.DataFrame()

print(titanic.head())
formula = 'survived ~ C(pclass) + C(sex) + age'
titanic_survival = glm(formula=formula, data=titanic, family=sm.families.Binomial())

titanic_survival_fit = titanic_survival.fit()
print(titanic_survival_fit.summary())
model_predictions['survival'] = titanic_survival_fit.predict(titanic)

# %%
# ## Question 3  
# Interpret your result. What are the factors and how do they affect the chance of survival (or the survival odds ratio)? What is the predicted probability of survival for a 30-year-old female with a second class ticket, no siblings, 3 parents/children on the trip? Use whatever variables that are relevant in your model.  
# The pseudo R-square value is 0.3438, which in itself is not too promising for the model;
# ideally we would like a higher r-squared value, as it would indicate that we could
# better predict the chances of survival from just the data features. However, this does
# make sense as there would be a lot of chance and other factors involved with surviving 
# the titanic crash.

# The deviance of the model was 486.06 (or negative two times Log-Likelihood-function)
# df = 398 
deviance = -2*titanic_survival_fit.llf
# Compare to the null deviance
null_deviance = titanic_survival_fit.null_deviance
print('Difference between deviance and null deviance: ', (null_deviance - deviance))
# difference = 369 with 4 variables. This would seem to indicate that the model is
# fairly good, as there is a large difference with a small amount of variables (i.e. they are
# significant)

#                      coef    std err          z      P>|z|      [0.025      0.975]
# ----------------------------------------------------------------------------------
# Intercept          2.8255      0.286      9.864      0.000       2.264       3.387
# C(pclass)[T.2]    -0.9289      0.248     -3.743      0.000      -1.415      -0.442
# C(pclass)[T.3]    -2.1722      0.236     -9.214      0.000      -2.634      -1.710
# C(sex)[T.male]    -2.6291      0.185    -14.212      0.000      -2.992      -2.267
# age               -0.0161      0.005     -3.018      0.003      -0.027      -0.006
# ==================================================================================

# The most important factors seems to be sex (being male decreases chance of survival),
# and the pclass (pclass of 3 has lower chance of survival).

# 

# %%
# ## Question 4  
# Try three different cut-off values at 0.3, 0.5, and 0.7. What are the a) Total accuracy of the model b) The precision of the model (average for 0 and 1), and c) the recall rate of the model (average for 0 and 1)
cut_offs = [0.3, 0.5, 0.7]

for cut_off in cut_offs:
    # Compute class predictions
    model_predictions['survival_' + str(cut_off)] = np.where(model_predictions['survival'] > cut_off, 1, 0)
    #
    # Make a cross table
    confusion: pd.DataFrame = pd.crosstab(titanic.survived, model_predictions['survival_' + str(cut_off)],
    rownames=['Actual'], colnames=['Predicted'],
    margins = True)

    # print(confusion)

    true_neg = confusion[0][0]
    false_pos = confusion[1][0]
    false_neg = confusion[0][1]
    true_pos = confusion[1][1]

    total = confusion['All']['All']
    accuracy = (confusion.iloc[1][1] + confusion.iloc[0][0]) / total
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    f1 = 2 * (precision*recall) / (precision + recall)
    print(f'Cutoff = {cut_off}:')
    print('f1: ', round(f1, 3))
    print('precision: ', round(precision, 3))
    print('recall: ', round(recall, 3))
    print()

#%%[markdown]
# # Part II  
# NFL field goal dataset - SciKitLearn
# 
# | Variable | Definition | Key/Notes  |  
# | ---- | ---- | ---- |   
# | AwayTeam | Name of visiting team | |  
# | HomeTeam | Name of home team | |  
# | qtr | quarter | 1, 2, 3, 4 |  
# | min | Time: minutes in the game |  |  
# | sec | Time: seconds in the game |  |  
# | kickteam | Name of kicking team |  |  
# | distance | Distance of the kick, from goal post (yards) |  |  
# | timerem | Time remaining in game (seconds) |  |  
# | GOOD | Whether the kick is good or no good | If not GOOD: |  
# | Missed | If the kick misses the mark | either Missed |  
# | Blocked | If the kick is blocked by the defense | or blocked |  
# 
# Part II
nfl = dm.api_dsLand('nfl2008_fga')
nfl.dropna(inplace=True)

#%% 
# ## Question 5  
# With the nfl dataset, perform some summary visualizations.  
under_60 = nfl[nfl['timerem'] < 60]['GOOD'].value_counts()
over_60 = nfl[nfl['timerem'] > 60]['GOOD'].value_counts()

print('Percentage of kicks made with less than 60 seconds left in the game: ', 
      round(under_60[1] / (under_60[1] + under_60[0]), 3))
print('Percentage of kicks made with more than 60 seconds left in the game: ', 
      round(over_60[1] / (over_60[1] + over_60[0]), 3))

# %%
plt.title('Kicks made at home versus away')
sns.histplot(data=nfl, x='homekick', hue='GOOD', multiple='dodge', shrink=0.8)

# %%
plt.title('Kicks made in each quarter')
sns.histplot(data=nfl, x='qtr', hue='GOOD', multiple='dodge', shrink=0.8)

# %%
plt.title('Kicks made over distance')
sns.lineplot(data=nfl, x='distance', y='GOOD', linewidth=2.5)


# %%
# 
# ## Question 6  
# Using the SciKitLearn library, build a logistic regression model overall (not individual team or kicker) to predict the chances of a successful field goal. What variables do you have in your model? 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

x_df = nfl[['qtr', 'distance', 'homekick', 'offscore', 'timerem']]
y_df = nfl[['GOOD']]

dietLogit = LogisticRegression()  # instantiate
x_trainDiet, x_testDiet, y_trainDiet, y_testDiet = train_test_split(x_df, y_df, random_state=1234 )
dietLogit.fit(x_trainDiet, y_trainDiet.values.ravel())
print('Logit model accuracy (with the test set):', dietLogit.score(x_testDiet, y_testDiet))
print('Logit model accuracy (with the train set):', dietLogit.score(x_trainDiet, y_trainDiet))

print("\nReady to continue.")

# The variables in my model are: the quarter the field goal was attempted, the distance
# it was attempted from, if the kick was at home or not, the offense score, and the time 
# remaining on the clock.
# The 'Missed' and 'Blocked' column felt like cheating, because combined they can indicate with
# 100% accuracy if the kick was good or not, so these features were ommitted.

# %%
# 
# ## Question 7  
# Someone has a feeling that home teams are more relaxed and have a friendly crowd, they should kick better field goals. Can you build two different models, one for all home teams, and one for road teams, of their chances of making a successful field goal?
# 
home = nfl[nfl.homekick == 1]
x_home = home[['qtr', 'distance', 'timerem']]
y_home = home[['GOOD']]

homeLogit = LogisticRegression()  # instantiate
x_trainDiet, x_testDiet, y_trainDiet, y_testDiet = train_test_split(x_home, y_home, random_state=1234 )
homeLogit.fit(x_trainDiet, y_trainDiet.values.ravel())
test = homeLogit.predict_proba(x_testDiet)
print('home team accuracy: ', test[:, 1].mean())


away = nfl[nfl.homekick == 0]
x_away = away[['qtr', 'distance', 'timerem']]
y_away = away[['GOOD']]

awayLogit = LogisticRegression()  # instantiate
x_trainDiet, x_testDiet, y_trainDiet, y_testDiet = train_test_split(x_away, y_away, random_state=1234 )
awayLogit.fit(x_trainDiet, y_trainDiet.values.ravel())
test = awayLogit.predict_proba(x_testDiet)
print('Away team accuracy: ', test[:, 1].mean())

# ## Question 8    
# From what you found, do home teams and road teams have different chances of making a successful field goal? If one does, is that true for all distances, or only with a certain range?
# 
# I found that away teams have a slightly higher chance of making a field goal, at about 89.21% (instead 
# of the 84.97% for home teams)

