# Exercises

import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from statsmodels.formula.api import ols
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.feature_selection import f_regression 
from math import sqrt
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')

# Load the tips dataset from either pydataset or seaborn.
from pydataset import data

df = data('tips')


# Fit a linear regression model (ordinary least squares) and compute yhat, predictions of tip using total_bill. 
# You may follow these steps to do that:
# import the method from statsmodels: from statsmodels.formula.api import ols
# fit the model to your data, where x = total_bill and y = tip: regr = ols('y ~ x', data=df).fit()
# compute yhat, the predictions of tip using total_bill: df['yhat'] = regr.predict(df.x)



from statsmodels.formula.api import ols

df = df.rename(columns={'total_bill' : 'x', 'tip' : 'y'})

df = df[['x','y']]

regr = ols('y ~ x', data=df).fit()

df['yhat'] = regr.predict(df.x)


# Create a file evaluate.py that contains the following functions.
# Write a function, plot_residuals(x, y, dataframe) that takes the feature, the target, and the dataframe 
# as input and returns a residual plot. (hint: seaborn has an easy way to do this!)

x = 'x' 
y = 'y'
dataframe = df

df

def plot_residuals(x, y, dataframe):

    return sns.residplot(x=x,y=y,data=dataframe)

# Write a function, regression_errors(y, yhat), that takes in y and yhat, 
# returns the sum of squared errors (SSE), 
# explained sum of squares (ESS), 
# total sum of squares (TSS), 
# mean squared error (MSE) and 
# root mean squared error (RMSE).

y = df.y
yhat = df.yhat


def regression_errors(y, yhat):

    SSE = sum(yhat)
    ESS = sum((df.yhat - df.y.mean())**2)
    TSS = ESS + SSE
    MSE = SSE/len(yhat)
    RMSE = sqrt(MSE)

    return SSE, ESS, TSS, MSE, RMSE

regression_errors(y,yhat)

# Write a function, baseline_mean_errors(y), that takes in your target, y, 
# computes the SSE, MSE & RMSE 
# when yhat is equal to the mean of all y, and returns the error values (SSE, MSE, and RMSE).

y = df.y



def baseline_mean_errors(y):

    df['yhat_base'] = y.mean()

    SSE_baseline = sum(df.yhat_base)
    MSE_baseline = SSE_baseline/len(df.yhat_base)
    RMSE_baseline = sqrt(MSE_baseline)

    return SSE_baseline, MSE_baseline, RMSE_baseline



# Write a function, better_than_baseline(SSE), that returns true if your model performs better than the baseline, 
# otherwise false.

def better_than_baseline(y,yhat):

    SSE = sum(yhat)
    ESS = sum((yhat - df.y.mean())**2)
    TSS = ESS + SSE
    R2_test = ESS/TSS

    df['yhat_base'] = y.mean()

    SSE_base = sum(df.yhat_base)
    ESS_base = sum((df.yhat_base - df.y.mean())**2)
    TSS_base = ESS_base + SSE_base
    R2_base = ESS_base/TSS_base

    print(f"test R-squared: {R2_test}")
    print(f"base R-squared: {R2_base}")

    return R2_test > R2_base

better_than_baseline(y,yhat)

# Write a function, model_significance(ols_model), that takes the ols model as input and returns the amount of variance explained in your model,
# and the value telling you whether the correlation between the model and the tip value are statistically significant.



df = df[['x','y']]

df['yhat'] = regr.predict(df.x)

ols_model = ols('y ~ x', data=df).fit()


def model_significance(ols_model):

    R2 = ols_model.rsquared

    P = ols_model.f_pvalue

    return f"The amount of variance explained in this modle is {round(R2, 3)} and the Pvalue for the modle is {round(P, 6)}."

model_significance(ols_model)