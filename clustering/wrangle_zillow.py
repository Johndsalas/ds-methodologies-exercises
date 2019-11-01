import pandas as pd
import env
import scipy.stats as stats
import numpy as np
from sklearn.impute import SimpleImputer
import acquire as a

# acquire

def get_connection(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'


def get_zillow_data():
    query = '''
    SELECT p_17.parcelid, p_17.logerror, p_17.transactiondate, p.*

    FROM predictions_2017 p_17

    JOIN (SELECT parcelid, Max(transactiondate) as tdate
      
          FROM predictions_2017
        
          GROUP BY parcelid )as sq1

    ON (sq1.parcelid=p_17.parcelid and sq1.tdate = p_17.transactiondate )
    
    LEFT JOIN properties_2017 p 
    ON p_17.parcelid=p.parcelid

    LEFT JOIN storytype
    USING (storytypeid)

    LEFT JOIN airconditioningtype
    USING (airconditioningtypeid)

    LEFT JOIN architecturalstyletype
    USING (architecturalstyletypeid)

    LEFT JOIN buildingclasstype
    USING (buildingclasstypeid)

    LEFT JOIN heatingorsystemtype
    USING (heatingorsystemtypeid)

    LEFT JOIN propertylandusetype
    USING (propertylandusetypeid)

    LEFT JOIN typeconstructiontype
    USING (typeconstructiontypeid)

    WHERE (p.latitude is not null and p.longitude is not null)
    
    AND propertylandusedesc in ('Residential General','Single Family Residential', 'Rural Residence', 'Mobile Home', 'Bungalow', 'Manifactured, Modular, Prefabricated Homes', 'Patio Home', 'Inferred Single Family Residential');
    '''
    return pd.read_sql(query, get_connection('zillow'))

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


# Preperation

def handle_missing_values(df, prop_required_column = .5, prop_required_row = .75):
    threshold = int(round(prop_required_column*len(df.index),0))
    df.dropna(axis=1, thresh=threshold, inplace=True)
    threshold = int(round(prop_required_row*len(df.columns),0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df

def remove_columns(df, cols_to_remove):
    df = df.drop(columns=cols_to_remove)
    return df

def imputer_9000(df,columns,strategy):

    for column in columns:

        imputer = SimpleImputer(strategy = strategy)
        imputer = imputer.fit(df[[column]])
        df[[column]] = imputer.transform(df[[column]])

    return df

def zillow_clean():

    df = get_zillow_data()

    df = handle_missing_values(df, prop_required_column = .5, prop_required_row = .75)

    cols_to_remove = ['buildingqualitytypeid','heatingorsystemtypeid','propertyzoningdesc','censustractandblock']
    df = remove_columns(df, cols_to_remove)

    columns = ['calculatedbathnbr','fullbathcnt','regionidcity','regionidzip','unitcnt','yearbuilt']
    strategy = 'most_frequent'
    df = imputer_9000(df,columns,strategy)

    columns = ['calculatedbathnbr', 'calculatedfinishedsquarefeet','finishedsquarefeet12','lotsizesquarefeet','structuretaxvaluedollarcnt','taxvaluedollarcnt','taxamount','landtaxvaluedollarcnt']
    strategy = 'most_frequent'
    df = imputer_9000(df,columns,strategy)

    return df