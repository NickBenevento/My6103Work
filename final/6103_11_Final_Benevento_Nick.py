# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%%
import copy
import numpy as np
import pandas as pd
import dm6103 as dm

world1 = dm.api_dsLand('World1', 'id')
world2 = dm.api_dsLand('World2', 'id')

print("\nReady to continue.")

#%% [markdown]
# # Two Worlds (Continuation from midterm: Part I - 25%)
# 
# In the (midterm) mini-project, we used statistical tests and visualization to 
# studied these two worlds. Now let us use the modeling techniques we now know
# to give it another try. 
# 
# Use appropriate models that we learned in this class or elsewhere, 
# elucidate what these two world looks like. 
# 
# Having an accurate model (or not) however does not tell us if the worlds are 
# utopia or not. Is it possible to connect these concepts together? (Try something called 
# "feature importance"?)
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
# %%
# World 1: 
from statsmodels.formula.api import ols
import statsmodels.api as sm

formula = 'income00 ~ age00 + education + C(marital) + C(gender) + C(ethnic) + C(industry)'

income = ols(formula=formula, data=world1, family=sm.families.Binomial())

income_fit = income.fit()
print(income_fit.summary())
#                           OLS Regression Results                            
# ==============================================================================
# Dep. Variable:               income00   R-squared:                       0.848
# Model:                            OLS   Adj. R-squared:                  0.848
# Method:                 Least Squares   F-statistic:                     8908.
# Date:                Fri, 06 May 2022   Prob (F-statistic):               0.00
# Time:                        15:26:57   Log-Likelihood:            -2.5731e+05
# No. Observations:               24000   AIC:                         5.147e+05
# Df Residuals:                   23984   BIC:                         5.148e+05
# Df Model:                          15                                         
# Covariance Type:            nonrobust                                         
# ====================================================================================
#                        coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------------
# Intercept         2.866e+04    593.931     48.258      0.000    2.75e+04    2.98e+04
# C(marital)[T.1]     -1.6473    158.002     -0.010      0.992    -311.341     308.046
# C(marital)[T.2]   -240.2771    257.433     -0.933      0.351    -744.862     264.308
# C(marital)[T.3]   -263.4030    325.858     -0.808      0.419    -902.105     375.299
# C(gender)[T.1]    -103.9987    152.672     -0.681      0.496    -403.245     195.248
# C(ethnic)[T.1]      64.5741    175.292      0.368      0.713    -279.009     408.157
# C(ethnic)[T.2]      51.4882    185.872      0.277      0.782    -312.833     415.809
# C(industry)[T.1]  7772.7147    276.516     28.109      0.000    7230.726    8314.703
# C(industry)[T.2]  1.733e+04    288.246     60.113      0.000    1.68e+04    1.79e+04
# C(industry)[T.3]  2.489e+04    281.551     88.420      0.000    2.43e+04    2.54e+04
# C(industry)[T.4]  3.388e+04    328.164    103.234      0.000    3.32e+04    3.45e+04
# C(industry)[T.5]  3.971e+04    299.958    132.383      0.000    3.91e+04    4.03e+04
# C(industry)[T.6]  6.656e+04    289.639    229.802      0.000     6.6e+04    6.71e+04
# C(industry)[T.7]  8.518e+04    331.939    256.612      0.000    8.45e+04    8.58e+04
# age00              -10.6400      8.567     -1.242      0.214     -27.432       6.152
# education            9.1165     24.515      0.372      0.710     -38.935      57.168
# ==============================================================================

# %%
# World 2:
formula = 'income00 ~ age00 + education + C(marital) + C(gender) + C(ethnic) + C(industry)'

income = ols(formula=formula, data=world2, family=sm.families.Binomial())

income_fit = income.fit()
print(income_fit.summary())
# 
#                             OLS Regression Results                            
# ==============================================================================
# Dep. Variable:               income00   R-squared:                       0.846
# Model:                            OLS   Adj. R-squared:                  0.846
# Method:                 Least Squares   F-statistic:                     8779.
# Date:                Fri, 06 May 2022   Prob (F-statistic):               0.00
# Time:                        15:31:23   Log-Likelihood:            -2.5754e+05
# No. Observations:               24000   AIC:                         5.151e+05
# Df Residuals:                   23984   BIC:                         5.152e+05
# Df Model:                          15                                         
# Covariance Type:            nonrobust                                         
# ====================================================================================
#                        coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------------
# Intercept          2.96e+04    602.971     49.087      0.000    2.84e+04    3.08e+04
# C(marital)[T.1]    -50.7526    159.835     -0.318      0.751    -364.039     262.534
# C(marital)[T.2]    -48.9602    263.130     -0.186      0.852    -564.711     466.790
# C(marital)[T.3]   -409.4943    323.993     -1.264      0.206   -1044.541     225.552
# C(gender)[T.1]     -98.6305    144.511     -0.683      0.495    -381.881     184.620
# C(ethnic)[T.1]    -167.5269    175.056     -0.957      0.339    -510.648     175.595
# C(ethnic)[T.2]    -143.9631    175.248     -0.821      0.411    -487.460     199.533
# C(industry)[T.1]  7596.3002    278.871     27.239      0.000    7049.696    8142.905
# C(industry)[T.2]  1.722e+04    289.561     59.462      0.000    1.67e+04    1.78e+04
# C(industry)[T.3]  2.446e+04    281.027     87.043      0.000    2.39e+04     2.5e+04
# C(industry)[T.4]  3.413e+04    318.573    107.131      0.000    3.35e+04    3.48e+04
# C(industry)[T.5]  4.009e+04    293.799    136.459      0.000    3.95e+04    4.07e+04
# C(industry)[T.6]  6.619e+04    279.988    236.401      0.000    6.56e+04    6.67e+04
# C(industry)[T.7]  8.515e+04    323.875    262.916      0.000    8.45e+04    8.58e+04
# age00               -4.1694      8.664     -0.481      0.630     -21.152      12.813
# education          -64.8592     25.273     -2.566      0.010    -114.396     -15.322
# ==============================================================================

# %%
# As we can see from the above two feature coefficients, there are differences between the 
# worlds in what has an impact on income. In world 1, there are two ethnicities that have a large
# effect on income: C(ethnic)[T.1] = 64.5741, C(ethnic)[T.2] = 51.4882. However, in world 2, these 
# same ethnicites do not have a positive effect on predicting income. This is one indication that
# world1 is more biased, and generally not considered a utopia.

#%% [markdown]
#
# # Free Worlds (Continuation from midterm: Part II - 25%)
# 
# To-do: Complete the method/function predictFinalIncome towards the end of this Part II codes.  
#  
# The worlds are gifted with freedom. Sort of.  
# I have a model built for them. It predicts their MONTHLY income/earning growth, 
# base on the characteristics of the individual. You task is to first examine and 
# understand the model. If you don't like it, build you own world and own model. 
# For now, please help me finish the last piece.  
# 
# My model will predict what is the growth factor for each person in the immediate month ahead. 
# Along the same line, it also calculate what is the expected (average) salary after 1 month with 
# that growth rate. You need to help make it complete, by producing a method/function that will 
# calculate what is the salary after n months. (Method: predictFinalIncome )  
# 
# That's all. Then try this model on people like Plato, and also create some of your favorite 
# people with all sort of different demographics, and see what their growth rates / growth factors 
# are in my worlds. Use the sample codes after the class definition below.  
# 
#%%
class Person:
  """ 
  a person with properties in the utopia 
  """

  def __init__(self, personinfo):
    self.age00 = personinfo['age'] # age at creation or record. Do not change.
    self.age = personinfo['age'] # age at current time. 
    self.income00 = personinfo['income'] # income at creation or record. Do not change.
    self.income = personinfo['income'] # income at current time.
    self.education = personinfo['education']
    self.gender = personinfo['gender']
    self.marital = personinfo['marital']
    self.ethnic = personinfo['ethnic']
    self.industry = personinfo['industry']
    # self.update({'age00': self.age00, 
    #         'age': self.age,
    #         'education': self.education,
    #         'gender': self.gender,
    #         'ethnic': self.ethnic,
    #         'marital': self.marital,
    #         'industry': self.industry,
    #         'income00': self.income00,
    #         'income': self.income})
    return
  
  def update(self, updateinfo):
    for key,val in updateinfo.items():
      if key in self.__dict__ : 
        self.__dict__[key] = val
    return
        
  def __getitem__(self, item):  # this will allow both person.gender or person["gender"] to access the data
    return self.__dict__[item]

  
#%%  
class myModel:
  """
  The earning growth model for individuals in the utopia. 
  This is a simplified version of what a model could look like, at least on how to calculate predicted values.
  """

  # ######## CONSTRUCTOR  #########
  def __init__(self, bias) :
    """
    :param bias: we will use this potential bias to explore different scenarios to the functions of gender and ethnicity

    :param b_0: the intercept of the model. This is like the null model. Or the current average value. 

    :param b_age: (not really a param. it's more a function/method) if the model prediction of the target is linearly proportional to age, this would the constant coefficient. In general, this does not have to be a constant, and age does not even have to be numerical. So we will treat this b_age as a function to convert the value (numerical or not) of age into a final value to be combined with b_0 and the others 
    
    :param b_education: similar. 
    
    :param b_gender: similar
    
    :param b_marital: these categorical (coded into numeric) levels would have highly non-linear relationship, which we typically use seaparate constants to capture their effects. But they are all recorded in this one function b_martial
    
    :param b_ethnic: similar
    
    :param b_industry: similar
    
    :param b_income: similar. Does higher salary have higher income or lower income growth rate as lower salary earners?
    """

    self.bias = bias # bias is a dictionary with info to set bias on the gender function and the ethnic function

    # ##################################################
    # The inner workings of the model below:           #
    # ##################################################

    self.b_0 = 0.0023 # 0.23% MONTHLY grwoth rate as the baseline. We will add/subtract from here

    # Technically, this is the end of the constructor. Don't change the indent

  # The rest of the "coefficients" b_1, b_2, etc are now disguised as functions/methods
  def b_age(self, age): # a small negative effect on monthly growth rate before age 45, and slight positive after 45
    effect = -0.00035 if (age<40) else 0.00035 if (age>50) else 0.00007*(age-45)
    return effect

  def b_education(self, education): 
    effect = -0.0006 if (education < 8) else -0.00025 if (education <13) else 0.00018 if (education <17) else 0.00045 if (education < 20) else 0.0009
    return effect

  def b_gender(self, gender):
    effect = 0
    biasfactor = 1 if ( self.bias["gender"]==True or self.bias["gender"] > 0) else 0 if ( self.bias["gender"]==False or self.bias["gender"] ==0 ) else -1  # for bias, no-bias, and reverse bias
    effect = -0.00045 if (gender<1) else 0.00045  # This amount to about 1% difference annually
    return biasfactor * effect 

  def b_marital(self, marital): 
    effect = 0 # let's assume martial status does not affect income growth rate 
    return effect

  def b_ethnic(self, ethnic):
    effect = 0
    biasfactor = 1 if ( self.bias["ethnic"]==True or self.bias["ethnic"] > 0) else 0 if ( self.bias["ethnic"]==False or self.bias["ethnic"] ==0 ) else -1  # for bias, no-bias, and reverse bias
    effect = -0.0006 if (ethnic < 1) else -0.00027 if (ethnic < 2) else 0.00045 
    return biasfactor * effect

  def b_industry(self, industry):
    effect = 0 if (industry < 2) else 0.00018 if (industry <4) else 0.00045 if (industry <5) else 0.00027 if (industry < 6) else 0.00045 if (industry < 7) else 0.00055
    return effect

  def b_income(self, income):
    # This is the kicker! 
    # More disposable income allow people to invest (stocks, real estate, bitcoin). Average gives them 6-10% annual return. 
    # Let us be conservative, and give them 0.6% return annually on their total income. So say roughly 0.0005 each month.
    # You can turn off this effect and compare the difference if you like. Comment in-or-out the next two lines to do that. 
    # effect = 0
    effect = 0 if (income < 50000) else 0.0001 if (income <65000) else 0.00018 if (income <90000) else 0.00035 if (income < 120000) else 0.00045 
    # Notice that this is his/her income affecting his/her future income. It's exponential in natural. 
    return effect

    # ##################################################
    # end of black box / inner structure of the model  #
    # ##################################################

  # other methods/functions
  def predictGrowthFactor( self, person ): # this is the MONTHLY growth FACTOR
    factor = 1 + self.b_0 + self.b_age( person["age"] ) + self.b_education( person['education'] ) + self.b_ethnic( person['ethnic'] ) + self.b_gender( person['gender'] ) + self.b_income( person['income'] ) + self.b_industry( person['industry'] ) + self.b_marital( ['marital'] )
    # becareful that age00 and income00 are the values of the initial record of the dataset/dataframe. 
    # After some time, these two values might have changed. We should use the current values 
    # for age and income in these calculations.
    return factor

  def predictIncome( self, person ): # perdict the new income one MONTH later. (At least on average, each month the income grows.)
    return person['income']*self.predictGrowthFactor( person )

  def predictFinalIncome( self, n, person ):
    # predict final income after n months from the initial record.
    # the right codes should be no longer than a few lines.
    # If possible, please also consider the fact that the person is getting older by the month. 
    # The variable age value keeps changing as we progress with the future prediction.
    # create a copy of the person for prediction
    # prediction = copy.deepcopy(person)

    for i in range(n):
      # update the income and age of the person
      person.update({'income': self.predictIncome(person), 'age': person['age'] + n/12})
      # prediction.update({'income': self.predictIncome(prediction), 'age': prediction['age'] + 1/12})
    # return prediction['income']
    return person['income']



print("\nReady to continue.")

#%%
# SAMPLE CODES to try out the model
utopModel = myModel( { "gender": False, "ethnic": False } ) # no bias Utopia model
biasModel = myModel( { "gender": True, "ethnic": True } ) # bias, flawed, real world model

print("\nReady to continue.")

#%%
# Now try the two models on some versions of different people. 
# See what kind of range you can get. Plato is here for you as an example.
# industry: 0-leisure n hospitality, 1-retail , 2- Education 17024, 3-Health, 4-construction, 5-manufacturing, 6-professional n business, 7-finance
# gender: 0-female, 1-male
# marital: 0-never, 1-married, 2-divorced, 3-widowed
# ethnic: 0, 1, 2 
# age: 30-60, although there is no hard limit what you put in here.
# income: no real limit here.

months = 12 # Try months = 1, 12, 60, 120, 360
# In the ideal world model with no bias
plato = Person( { "age": 58, "education": 20, "gender": 1, "marital": 0, "ethnic": 2, "industry": 7, "income": 100000 } )
print(f'plato growth factor: {utopModel.predictGrowthFactor(plato)}') # This is the current growth factor for plato
print(f'plato income 1 month later: {utopModel.predictIncome(plato)}') # This is the income after 1 month
# Do the following line when your new function predictFinalIncome is ready
print(f'12 months: {utopModel.predictFinalIncome(months,plato)}')
#
# If plato ever gets a raise, or get older, you can update the info with a dictionary:
# plato.update( { "age": 59, "education": 21, "marital": 1, "income": 130000 } )

# In the flawed world model with biases on gender and ethnicity 
aristotle = Person( { "age": 58, "education": 20, "gender": 1, "marital": 0, "ethnic": 2, "industry": 7, "income": 100000 } )
print(f'bias: {biasModel.predictGrowthFactor(aristotle)}') # This is the current growth factor for aristotle
print(f'bias: {biasModel.predictIncome(aristotle)}') # This is the income after 1 month
# Do the following line when your new function predictFinalIncome is ready
print(f'bias aristotel predicted income after 12 months: {biasModel.predictFinalIncome(months,aristotle)}')

print("\nReady to continue.")

# %%
# John, a single male, finance major
john = Person( { "age": 30, "education": 15, "gender": 1, "marital": 0, "ethnic": 0, "industry": 7, "income": 120000 } )
print(f'John bias growth: {biasModel.predictGrowthFactor(john)}') # This is the current growth factor for aristotle
print(f'John bias predicted income after 12 months: {biasModel.predictFinalIncome(months, john)}')
john = Person( { "age": 30, "education": 15, "gender": 1, "marital": 0, "ethnic": 0, "industry": 7, "income": 120000 } )
print(f'John utop growth: {utopModel.predictGrowthFactor(john)}') # This is the current growth factor for aristotle
print(f'John utop predicted income after 12 months: {utopModel.predictFinalIncome(months, john)}')
# The bias model predicts an income of 124370.92, while the utop model predicts 124594.30

# Mary, a married woman, education major
mary = Person( { "age": 45, "education": 17, "gender": 0, "marital": 1, "ethnic": 1, "industry": 2, "income": 95000 } )
print(f'Mary bias growth: {biasModel.predictGrowthFactor(mary)}') # This is the current growth factor for aristotle
print(f'Mary bias predicted income after 12 months: {biasModel.predictFinalIncome(months, mary)}')
mary = Person( { "age": 45, "education": 17, "gender": 0, "marital": 1, "ethnic": 1, "industry": 2, "income": 95000 } )
print(f'Mary utop growth: {utopModel.predictGrowthFactor(mary)}') # This is the current growth factor for aristotle
print(f'Mary utop predicted income after 12 months: {utopModel.predictFinalIncome(months, mary)}')
# The bias model predicts and income of 98268.06, while the utop model predicts 99118.06

# Joe, an older widower, manufactoring major
joe = Person( { "age": 60, "education": 20, "gender": 1, "marital": 3, "ethnic": 2, "industry": 5, "income": 105000 } )
print(f'Joe bias growth: {biasModel.predictGrowthFactor(joe)}') # This is the current growth factor for aristotle
print(f'Joe bias predicted income after 12 months: {biasModel.predictFinalIncome(months, joe)}')
joe = Person( { "age": 60, "education": 20, "gender": 1, "marital": 3, "ethnic": 2, "industry": 5, "income": 105000 } )
print(f'Joe utop growth: {utopModel.predictGrowthFactor(joe)}') # This is the current growth factor for aristotle
print(f'Joe utop predicted income after 12 months: {utopModel.predictFinalIncome(months, joe)}')
# The bias model predicts and income of 111569.38, while the utop model predicts 110376.40

#%% [markdown]
# # Evolution (Part III - 25%)
# 
# We want to let the 24k people in WORLD#2 to evolve, for 360 months. You can either loop them through, and 
# create a new income or incomeFinal variable in the dataframe to store the new income level after 30 years. Or if you can figure out a way to do 
# broadcasting the predict function on the entire dataframem that can work too. If you loop through them, you can also consider 
# using Person class to instantiate the person and do the calcuations that way, then destroy it when done to save memory and resources. 
# If the person has life changes, it's much easier to handle it that way, then just tranforming the dataframe directly.
# 
# We have just this one goal, to see what the world look like after 30 years, according to the two models (utopModel and biasModel). 
# 
# Remember that in the midterm, world1 in terms of gender and ethnic groups, 
# there were not much bias. Now if we let the world to evolve under the 
# utopia model utopmodel, and the biased model biasmodel, what will the income distributions 
# look like after 30 years?
# 
# Answer this in terms of distribution of income only. I don't care about 
# other utopian measures in this question here. 
# 

months = 360
# rename cols for person class instantiation
world2 = world2.rename(columns={'income00': 'income', 'age00': 'age'})
# evolve the world
for i, row in world2.iterrows():
  world2.at[i, 'utopFinalIncome'] = utopModel.predictFinalIncome(months, Person(row))
  world2.at[i, 'biasFinalIncome'] = biasModel.predictFinalIncome(months, Person(row))

# %%
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn')

# %%
# view plots individually
sns.histplot(x=world2['utopFinalIncome'], kde = False)

# %%
sns.histplot(x=world2['biasFinalIncome'], kde = False)

# %%
# overlay the plots
sns.histplot(x=world2['utopFinalIncome'], kde = False)
sns.histplot(x=world2['biasFinalIncome'], kde = False, color='b', alpha=0.4)

# Look at mean and median income distributions
print('Utop model income mean: ', world2['utopFinalIncome'].mean())
print('Bias model income mean: ', world2['biasFinalIncome'].mean())

print('Utop model income median: ', world2['utopFinalIncome'].median())
print('Bias model income median: ', world2['biasFinalIncome'].median())
# Utop model income mean:  205858.19627635033
# Bias model income mean:  199969.74609538502
# Utop model income median:  172735.36600697602
# Bias model income median:  163887.31154910813

# The utopian model has higher mean and median income distributions

#%% 
# # Reverse Action (Part IV - 25%)
# 
# Now let us turn our attension to World 1, which you should have found in the midterm that 
# it is far from being fair from income perspective among gender and ethnic considerations. 
# 
# Let us now put in place some policy action to reverse course, and create a revser bias model:
revbiasModel = myModel( { "gender": -1, "ethnic": -1 } ) # revsered bias, to right what is wronged gradually.

# If we start off with Word 1 on this revbiasModel, is there a chance for the world to eventual become fair like World #2? If so, how long does it take, to be fair for the different genders? How long for the different ethnic groups? 

# If the current model cannot get the job done, feel free to tweak the model with more aggressive intervention to change the growth rate percentages on gender and ethnicity to make it work. 

months = 1
# Set equality threshold as $5,000
equality_threshold = 5000
# rename cols for person class instantiation
world1 = world1.rename(columns={'income00': 'income', 'age00': 'age'})
sex_equality = False
ethnicity_equality = False

# evolve the world
while months < 360:

  for i, row in world1.iterrows():
    world1.at[i, 'revbiasFinalIncome'] = revbiasModel.predictFinalIncome(months, Person(row))
    # Update the person's age
    world1.at[i, 'age'] = world1.at[i, 'age'] + 5/12
  

  income_male = world1[world1.gender == 1].loc[:, 'revbiasFinalIncome'].mean()
  income_female = world1[world1.gender == 0].loc[:, 'revbiasFinalIncome'].mean()
  if abs(income_male - income_female) < equality_threshold:
    print(f'Male and female incomes equal at {months} months')
    print('Average male income: ', income_male)
    print('Average female income: ', income_female)
    sex_equality = True


  income_e0 = world1[world1.ethnic == 0].loc[:, 'revbiasFinalIncome'].mean()
  income_e1 = world1[world1.ethnic == 1].loc[:, 'revbiasFinalIncome'].mean()
  income_e2 = world1[world1.ethnic == 2].loc[:, 'revbiasFinalIncome'].mean()

  if abs(income_e0 - income_e1) < equality_threshold \
    and abs(income_e1 - income_e2) < equality_threshold \
    and abs(income_e0 - income_e2) < equality_threshold:
    print(f'Ethnicity incomes equal at {months} months')
    print('Average ethnicity 0 income: ', income_e0)
    print('Average ethnicity 1 income: ', income_e1)
    print('Average ethnicity 2 income: ', income_e2)
    ethnicity_equality = True

  if sex_equality and ethnicity_equality:
    break


  months += 5

# The male and female incomes were equal and stayed equal after around 175 months.
# At 195 months, the average male income was 122239.16, and the 
# average female income was 118856.56, with a difference of 3,382.56.
# The incomes across the different ethnicities never became equal however, testing
# a time frame of up to 30 years.

# %%
# Final gender income differences after 30 years
men = world1[world1.gender == 1]
m_avg = men.loc[:, 'revbiasFinalIncome'].mean()
m_med = men.loc[:, 'revbiasFinalIncome'].median()
women = world1[world1.gender == 0]
w_avg = women.loc[:, 'revbiasFinalIncome'].mean()
w_med = women.loc[:, 'revbiasFinalIncome'].median()

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
#%%
# Final ethnicity income differences after 30 years
ethnicity1 = world1[world1.ethnic == 0]
e1_avg = ethnicity1.loc[:, 'revbiasFinalIncome'].mean()
e1_med = ethnicity1.loc[:, 'revbiasFinalIncome'].median()

ethnicity2 = world1[world1.ethnic == 1]
e2_avg = ethnicity2.loc[:, 'revbiasFinalIncome'].mean()
e2_med = ethnicity2.loc[:, 'revbiasFinalIncome'].median()

ethnicity3 = world1[world1.ethnic == 2]
e3_avg = ethnicity3.loc[:, 'revbiasFinalIncome'].mean()
e3_med = ethnicity3.loc[:, 'revbiasFinalIncome'].median()

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
# %%
