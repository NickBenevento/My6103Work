#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import dm6103 as dm
plt.style.use('classic')

# Part I
titanic = dm.api_dsLand('Titanic', 'id')

# Part II
nfl = dm.api_dsLand('nfl2008_fga')
nfl.dropna(inplace=True)

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

# ## Question 2  
# Build a logistic regression model for survival using the statsmodels library. As we did before, include the features that you find plausible. Make sure categorical variables are use properly. If the coefficient(s) turns out insignificant, drop it and re-build.  
import statsmodels.api as sm  # Importing statsmodels
from statsmodels.formula.api import glm

model_predictions = pd.DataFrame()

print(titanic.head())
# formula = 'survived ~ pclass + sex + age'
# formula = 'survived ~ pclass + C(sex) + C(embarked)'
# formula = 'survived ~ pclass + C(sex) + sibsp + parch + fare'
formula = 'survived ~ C(pclass) + C(sex) + fare + age + C(embarked)'
titanic_survival = glm(formula=formula, data=titanic, family=sm.families.Binomial())

titanic_survival_fit = titanic_survival.fit()
print(titanic_survival_fit.summary())
model_predictions['survival'] = titanic_survival_fit.predict(titanic)

# %%
# ## Question 3  
# Interpret your result. What are the factors and how do they affect the chance of survival (or the survival odds ratio)? What is the predicted probability of survival for a 30-year-old female with a second class ticket, no siblings, 3 parents/children on the trip? Use whatever variables that are relevant in your model.  
# The deviance of the model was 486.06 (or negative two times Log-Likelihood-function)
# df = 398 
print(-2*titanic_survival_fit.llf)
# Compare to the null deviance
print(titanic_survival_fit.null_deviance)

# test = np.array(['2', 'female', 30, 0, 3])
# print(type(test))
# test = pd.DataFrame(pclass='2', sex='female', 30, 0, 3)
# test = pd.DataFrame(columns=['pclass', 'sex', 'age', 'sibsp', 'parch', 'ticket', 'fare', 'embarked'],
#                     data=[2, 'female', 30, 0, 3, 123, 10, 'S'])
test = titanic.iloc[0].copy()
test['sex'] = 'female'
test['age'] = 30
test['sibsp'] = 0
test['parch'] = 3

print(titanic_survival.predict(test))
# 499.98  # df = 399 
# A decrease of 14 with just one variable. That's not bad. 
# 
# Another way to use the deviance value is to check the chi-sq p-value like this:
# Null model: chi-sq of 399.98, df = 399, the p-value is 0.000428 (can use scipy.stats.chisquare function) 
# Our model: chi-sq of 486.06, df = 398, the p-value is 0.001641
# These small p-values (less than 0.05, or 5%) means reject the null hypothesis, which means the model is not a good fit with data. We want higher p-value here. Nonetheless, the one-variable model is a lot better than the null model.

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
#%% 
# ## Question 5  
# With the nfl dataset, perform some summary visualizations.  
nfl.head()
# print(nfl.columns)
# for col in nfl.columns:
#     print(nfl[col].unique())
# print(nfl)

# %%
# 
# ## Question 6  
# Using the SciKitLearn library, build a logistic regression model overall (not individual team or kicker) to predict the chances of a successful field goal. What variables do you have in your model? 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# x_df = nfl.drop('GOOD', axis=1)
x_df = nfl[['qtr', 'distance', 'homekick', 'timerem']]
# x_df = nfl[['qtr', 'distance', 'homekick', 'timerem', 'Missed']]
# x_df = nfl[['Missed']]
y_df = nfl[['GOOD']]

dietLogit = LogisticRegression(max_iter=10000)  # instantiate
x_trainDiet, x_testDiet, y_trainDiet, y_testDiet = train_test_split(x_df, y_df, random_state=1234 )
dietLogit.fit(x_trainDiet, y_trainDiet.values.ravel())
print('Logit model accuracy (with the test set):', dietLogit.score(x_testDiet, y_testDiet))
print('Logit model accuracy (with the train set):', dietLogit.score(x_trainDiet, y_trainDiet))

print("\nReady to continue.")

# %%
# 
# ## Question 7  
# Someone has a feeling that home teams are more relaxed and have a friendly crowd, they should kick better field goals. Can you build two different models, one for all home teams, and one for road teams, of their chances of making a successful field goal?
# 
home = nfl[nfl.homekick == 1]
x_home = home[['qtr', 'distance', 'timerem']]
y_home = home[['GOOD']]

homeLogit = LogisticRegression(max_iter=10000)  # instantiate
x_trainDiet, x_testDiet, y_trainDiet, y_testDiet = train_test_split(x_home, y_home, random_state=1234 )
homeLogit.fit(x_trainDiet, y_trainDiet.values.ravel())
test = homeLogit.predict_proba(x_testDiet)
print(test[:, 1].mean())


away = nfl[nfl.homekick == 0]
x_away = away[['qtr', 'distance', 'timerem']]
y_away = away[['GOOD']]

awayLogit = LogisticRegression(max_iter=10000)  # instantiate
x_trainDiet, x_testDiet, y_trainDiet, y_testDiet = train_test_split(x_away, y_away, random_state=1234 )
awayLogit.fit(x_trainDiet, y_trainDiet.values.ravel())
test = awayLogit.predict_proba(x_testDiet)
print(test[:, 1].mean())

# ## Question 8    
# From what you found, do home teams and road teams have different chances of making a successful field goal? If one does, is that true for all distances, or only with a certain range?
# 


# %%
