#%%[markdown]
# You may use web search, notes, etc. 
# Do not use help from another human. If you use help from another student, 
# then I have no choice but to consider that student not a human, and will be 
# booted off my class immediately. You will also arrive at the same fate.
# 
#%%
import pandas as pd
import dm6103 as dm
df = dm.api_dsLand('Diet6wk','Person')
df.columns.values[3] = 'origweight'
df.info()

# The dataframe is on a person's weight 6 weeks after starting a diet. 
# Build these models:
# 
# 1. Using statsmodels library, build a linear model for the wight6weeks as a function of the other variables. Use gender and Diet as categorical variables. Print out the model summary. What is the r-squared value of the model?  
# 
from statsmodels.formula.api import ols
formula = 'weight6weeks ~ C(gender) + Age + Height + origweight + C(Diet)'
model_weight = ols(formula=formula, data=df)

model_weight_fit = model_weight.fit()
print(model_weight_fit.summary())
print('r-squared value: 0.930')


#%%
# 2. Again using the statsmodels library, build a multinomial-logit regression model for the Diet (3 levels) as a function of the other variables. Use gender as categorical again. Print out the model summary. What is the  model's "psuedo r-squared" value?  
# 
from statsmodels.formula.api import mnlogit  # use this for multinomial logit in statsmodels library, instead of glm for binomial.
formula = 'Diet ~ C(gender) + Age + Height + origweight + weight6weeks'
# Sample use/syntax:
model = mnlogit(formula, df)

model_fit = model.fit()
print(model_fit.summary())
print('Pseudo R-squared: 0.09026')



#%%
# 3a. Use SKLearn from here onwards. 
# Use a 2:1 split, set up the training and test sets for the dataset, with Diet as y, and the rest as Xs. Use the seed value/random state as 1234 for the split.
#
from sklearn import linear_model
from sklearn.model_selection import train_test_split
x_df = df[['gender', 'Age', 'Height', 'origweight', 'weight6weeks']]
y_df = df[['Diet']]

x_trainDiet, x_testDiet, y_trainDiet, y_testDiet = train_test_split(x_df, y_df, test_size=0.33, random_state=1234 )
full_split1 = linear_model.LinearRegression() # new instancew
full_split1.fit(x_trainDiet, y_trainDiet)
y_pred1 = full_split1.predict(x_testDiet)
full_split1.score(x_testDiet, y_testDiet)

print('score (train):', full_split1.score(x_trainDiet, y_trainDiet))
print('score (test):', full_split1.score(x_testDiet, y_testDiet))
print('intercept:', full_split1.intercept_)
print('coef_:', full_split1.coef_)

#%%
# 
# 3b. Build the corresponding logit regression as in Q2 here using sklearn. Train and score it. What is the score of your model with the training set and with the test set?
# 
from sklearn.linear_model import LogisticRegression

dietLogit = LogisticRegression(max_iter=10000)  # instantiate
x_trainDiet, x_testDiet, y_trainDiet, y_testDiet = train_test_split(x_df, y_df, random_state=1234 )
dietLogit.fit(x_trainDiet, y_trainDiet.values.ravel())
print('Logit model accuracy (with the test set):', dietLogit.score(x_testDiet, y_testDiet))
print('Logit model accuracy (with the train set):', dietLogit.score(x_trainDiet, y_trainDiet))

print("\nReady to continue.")


#%%
# 4. Using the same training dataset, now use a 3-NN model, score the model with the training and test datasets. 
# 
# 3-KNN algorithm
# The best way
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

knn_cv = KNeighborsClassifier(n_neighbors=3) # instantiate with n value given

knn_cv.fit(x_trainDiet, y_trainDiet.values.ravel())
ytest_pred = knn_cv.predict(x_testDiet)
print('Model score on test data: ', knn_cv.score(x_testDiet,y_testDiet))
print('Model score on train data: ', knn_cv.score(x_trainDiet,y_trainDiet))

#%%
