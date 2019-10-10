# Our scenario continues:
# As a customer analyst, I want to know who has spent the most money with us over their lifetime. I have monthly charges and tenure, so I think I will be able to use those two attributes as features to estimate total_charges. I need to do this within an average of $5.00 per customer.
# Create split_scale.py that will contain the functions that follow. Each scaler function should create the object, fit and transform both train and test. They should return the scaler, train dataframe scaled, test dataframe scaled. Be sure your indices represent the original indices from train/test, as those represent the indices from the original dataframe. Be sure to set a random state where applicable for reproducibility!
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer, RobustScaler, MinMaxScaler

import wrangle as w

df = w.wrangle_telco()

x = df[['total_charges']]

y = df[['tenure','monthly_charges']]

train_pct = .7

df = df.drop(columns='customer_id')

def split_my_data(x, y, train_pct):

    x_train, x_test, y_train, y_test =  train_test_split(x,y, train_size = train_pct, random_state = 999)

    return x_train, x_test, y_train, y_test


def split_data_whole(df):

    train, test = train_test_split(df, train_size = train_pct, random_state = 999)

    return train, test


def standard_scaler(train,test):

    scaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(train)

    train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])

    test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])

    return train_scaled, test_scaled, scaler


def scale_inverse(train_scaled,test_scaled):

        scaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(train)

        train_unscaled = pd.DataFrame(scaler.inverse_transform(train_scaled), columns=train_scaled.columns.values).set_index([train.index.values])
        test_unscaled = pd.DataFrame(scaler.inverse_transform(test_scaled), columns=test_scaled.columns.values).set_index([test.index.values])

        return train_unscaled, test_unscaled, scaler

train, test = split_data_whole(df)
print(train)
train_scaled, test_scaled = standard_scaler(train,test)
print(train_scaled)
train_unscaled, test_unscaled = scale_inverse(train_scaled,test_scaled)
print(train_unscaled)

def uniform_scaler(train,test):

    scaler = QuantileTransformer(n_quantiles=100, output_distribution='uniform', random_state=123, copy=True).fit(train)

    train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
    test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])

    return train_scaled, test_scaled, scaler


def gaussian_scaler(train):

    scaler = PowerTransformer(method='yeo-johnson', standardize=False, copy=True).fit(train)

    train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
    test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])

    return train_scaled, test_scaled, scaler


def min_max_scaler(df):

    scaler = MinMaxScaler(copy=True, feature_range=(0,1)).fit(train)

    train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])

    test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])

    return train_scaled, test_scaled, scaler


def iqr_robust_scaler(df):

    scaler = RobustScaler(quantile_range=(25.0,75.0), copy=True, with_centering=True, with_scaling=True).fit(train)

    train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])

    test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])

    return train_scaled, test_scaled, scaler

