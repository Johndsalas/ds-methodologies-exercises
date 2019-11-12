import pandas as pd
import env
import scipy.stats as stats
import numpy as np
from sklearn.impute import SimpleImputer
import acquire as a
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Acquire

def get_connection(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def get_mallcustomer_data():
    df = pd.read_sql('SELECT * FROM customers;', get_connection('mall_customers'))
    return df.set_index('customer_id')

# Summery

def nulls_by_col(df):
    num_missing = df.isnull().sum()
    rows = df.shape[0]
    pct_missing = num_missing/rows
    cols_missing = pd.DataFrame({'num_rows_missing': num_missing, 'pct_rows_missing': pct_missing})
    return cols_missing

def nulls_by_row(df):
    num_cols_missing = df.isnull().sum(axis=1)
    pct_cols_missing = df.isnull().sum(axis=1)/df.shape[1]*100
    rows_missing = pd.DataFrame({'num_cols_missing': num_cols_missing, 'pct_cols_missing': pct_cols_missing}).reset_index().groupby(['num_cols_missing','pct_cols_missing']).count().rename(index=str, columns={'index': 'num_rows'}).reset_index()
    return rows_missing

def df_value_counts(df):
    counts = pd.Series([])
    for i, col in enumerate(df.columns.values):
        if df[col].dtype == 'object':
            col_count = df[col].value_counts()
        else:
            col_count = df[col].value_counts(bins=10, sort=False)
        counts = counts.append(col_count)
    return counts

def df_summary(df):
    print('--- Shape: {}'.format(df.shape))
    print('--- Info')
    df.info()
    print('--- Descriptions')
    print(df.describe(include='all'))
    print('--- Nulls By Column')
    print(nulls_by_col(df))
    print('--- Nulls By Row')
    print(nulls_by_row(df))
    print('--- Value Counts')
    print(df_value_counts(df))

# Handle Outliers

def get_upper_outliers(s, k):
 
    q1, q3 = s.quantile([.25, .75])
    iqr = q3 - q1
    upper_bound = q3 + (k * iqr)
    return s.apply(lambda x: max([x - upper_bound, 0]))

# def get_lower_outliers(s,k):

#     q1, q3 = s.quantile([.25, .75])
#     iqr = q3 - q1
#     lower_bound = q1 - (k * iqr)
#     return s.apply(lambda x: max([x - lower_bound, 0]))

def add_outlier_columns(df, k):
    
    for col in df.select_dtypes('number'):
        df[col + '_outliers'] = get_upper_outliers(df[col], k)

    # for col in df.select_dtypes('number'):
    #     df[col + 'lower_outliers'] = get_lower_outliers(df[col], k)

    return df

# Split 

def split(df,train_pct=.7):

    train, test = train_test_split(df, train_size = train_pct, random_state = 999)

    return test, train

# Encode

def one_hot_encode(train, test, col_name):

    encoded_values = sorted(list(train[col_name].unique()))

    # create 2D np arrays of the encoded variable (in train and test)
    train_array = np.array(train[col_name]).reshape(len(train[col_name]),1)
    test_array = np.array(test[col_name]).reshape(len(test[col_name]),1)

    # One Hot Encoding
    ohe = OneHotEncoder(sparse=False, categories='auto')
    train_ohe = ohe.fit_transform(train_array)
    test_ohe = ohe.transform(test_array)

    # Turn the array of new values into a data frame with columns names being the values
    # and index matching that of train/test
    # then merge the new dataframe with the existing train/test dataframe
    train_encoded = pd.DataFrame(data=train_ohe,
                            columns=encoded_values, index=train.index)
    train = train.join(train_encoded)

    test_encoded = pd.DataFrame(data=test_ohe,
                               columns=encoded_values, index=test.index)
    test = test.join(test_encoded)

    return test, train

# No missing values

# Scaling

def standard_scaler(train,test):

    scaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(train)

    train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])

    test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])

    return train_scaled, test_scaled, scaler

def mall_clean():

    df = get_mallcustomer_data()

    k=1.5

    df = add_outlier_columns(df, k)
    test, train = split(df,train_pct=.7)

    col_name = 'gender'
    test, train, = one_hot_encode(train, test, col_name)
    test.drop(columns='gender',inplace=True)
    train.drop(columns='gender',inplace=True)

    train_scaled, test_scaled, scaler = standard_scaler(train,test)

    return train_scaled, test_scaled, scaler