# Exercises
# Our scenario continues:
# As a customer analyst, I want to know who has spent the most money with us over their lifetime. 
# I have monthly charges and tenure, so I think I will be able to use those two attributes as features to estimate total_charges. 
# I need to do this within an average of $5.00 per customer.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

import env
import wrangle as w
import split_scale as ss 

df = w.wrangle_telco()

x = df[['tenure','monthly_charges']]

y = df[['total_charges']]

x_train, x_test, y_train, y_test = ss.split_my_data(x, y, train_pct=.8)

# 1. Write a function, select_kbest_freg_unscaled() that takes X_train, y_train and k as input 
# (X_train and y_train should not be scaled!) and returns a list of the top k features.

from sklearn.feature_selection import SelectKBest, f_regression

k = 1

# print(x_train)
# print(y_train)

# print(x_test)
# print(y_test)



def select_kbest_freg_unscaled(x_train, y_train, k):

    f_selector = SelectKBest(f_regression, k=k).fit(x_train,y_train)

    f_support = f_selector.get_support()

    f_feature = x_train.loc[:,f_support].columns.tolist()

    print(str(len(f_feature)), 'selected features')
    print(f_feature)
    
select_kbest_freg_unscaled(x_train, y_train, k)

# 2. Write a function, select_kbest_freg_scaled() that takes X_train, y_train (scaled) and k as input
#  and returns a list of the top k features.

def standard_scaler(x_train):

    scaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(x_train)

    x_train_scaled = pd.DataFrame(scaler.transform(x_train), columns=x_train.columns.values).set_index([x_train.index.values])

    return x_train_scaled

x_train_scaled = standard_scaler(x_train)

k = 1

def select_kbest_freg_scaled(x_train_scaled, y_train, k):

    f_selector = SelectKBest(f_regression, k=k).fit(x_train,y_train)

    f_support = f_selector.get_support()

    f_feature = x_train.loc[:,f_support].columns.tolist()

    print(str(len(f_feature)), 'selected features')
    print(f_feature)

select_kbest_freg_scaled(x_train_scaled, y_train_scaled, k)
# 3. Write a function, ols_backwared_elimination() that takes X_train and y_train (scaled) as input and returns 
# selected features based on the ols backwards elimination method.

import statsmodels.api as sm

def ols_backwared_elimination(x_train_scaled,y_train):

    cols = list(x_train_scaled.columns)
    pmax = 1

    while (len(cols) > 0):
        p = []
        x_1 = x_train_scaled[cols]

        model = sm.OLS(y_train, x_1).fit()

        p = pd.Series(model.pvalues.values[0:,],index=cols)

        pmax = max(p)

        feature_with_pmax = p.idxmax()
   
        if(pmax>0.05):
            cols.remove(feature_with_pmax)
        else:
            break

    return cols

ols_backwared_elimination(x_train_scaled,y_train)

# Write a function, lasso_cv_coef() that takes X_train and y_train as input 
# and returns the coefficients for each feature, along with a plot of the features and their weights.

from sklearn.linear_model import LassoCV

def lasso_cv_coef(X_train,y_train):

    reg = LassoCV()
    reg.fit(X_train, y_train)

    coef = pd.Series(reg.coef_, index = X_train.columns)

    imp_coef = coef.sort_values()

    matplotlib.rcParams['figure.figsize'] = (4.0, 5.0)
    imp_coef.plot(kind = "barh")
    plt.title("Feature importance using Lasso Model")

    return coef

lasso_cv_coef(x_train,y_train)

# Write 3 functions, the first computes the number of optimum features (n) using rfe, 
# the second takes n as input and returns the top n features, and the 
# third takes the list of the top n features as input and returns a new X_train and X_test dataframe with those top features.

from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE

def recursive_feature_elimination(x_train,y_train):

    number_of_features_list=np.arange(1,3)
    high_score=0

    number_of_features=0
    score_list =[]

    for n in range(len(number_of_features_list)):
        model = LinearRegression()
        rfe = RFE(model,number_of_features_list[n])
        X_train_rfe = rfe.fit_transform(x_train,y_train)
        X_test_rfe = rfe.transform(x_test)
        model.fit(X_train_rfe,y_train)
        score = model.score(X_test_rfe,y_test)
        score_list.append(score)
        if(score>high_score):
            high_score = score
            number_of_features = number_of_features_list[n]

    return number_of_features

n = recursive_feature_elimination(x_train,y_train)