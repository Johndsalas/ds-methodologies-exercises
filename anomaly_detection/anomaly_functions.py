import pandas as import pd
import seaborn as sns
import matplotlib.pyplot as plt


def get_outliers(df):
    '''
    Function takes in a data set and returns a data set with columns as rows.
    Each row will show quantile information as well as upper bound and lowerbound outliers.
    mild outliers calculated with multiplier of 1.5 extream use multiplier of 3
    '''
    
    # get transformation of describe of df
    df = df.describe().T
    
    # dropp useless columns
    df = df.drop(columns=['mean', 'count', 'std'])
    
    # replace % in column names
    df.columns = df.columns.str.replace("[%]", "")
    
    # rename columns
    df =df.rename(columns={'25': 'Q1', '50': 'Q2', '75': 'Q3'})
    
    # add column calculating the interquantile range
    df['iqr']= df['Q3'] - df['Q1']
    
    # add add column multiplyer for mild and extream values
    df['mild']= 1.5
    df['extream']= 3
    
    # add columns calculating upper and lower 'mild' outlire values
    df['mild_upperbound']= df['Q3'] + (df.iqr*df.mild)
    df['mild_lowerbound']= df['Q1'] - (df.iqr*df.mild)
    
    # add columns calculating the upper and lower 'extream' outlier values
    df['extream_upperbound']= df['Q3'] + (df.iqr*df.extream)
    df['extream_lowerbound']= df['Q1'] - (df.iqr*df.extream)
    
    # dropp 'multiplyer' columns to unclutter data frame
    df = df.drop(columns=['mild','extream'])
    
    return df


def get_lower_and_upper_bounds(series, multiplier):

    '''
    Function takes in a series and a multiplier and returns the upper bound and lower bound for outliers in that series
    '''
    s = series
    
    iqr = s.quantile(.75) - s.quantile(.25)
    
    upper_bound = s.quantile(.75) + (iqr*multiplier)
    
    lower_bound = s.quantile(.25) - (iqr*multiplier)
    
    return upper_bound, lower_bound
   
def get_outlire_graph(df):
    '''
    Function takes in a dataframe and prints a graph of outliers at the 2 and 3 sigma level
    '''

    # create list of columns in the data frame that contain int or float values 
    lst = [column for column in df.columns if df[f'{column}'].dtype in ('int64','float64')]

    # loop through list and append a column to the datafram that identifies inliers at 2nd and 3rd sigma level
    for column in lst:

        df[f'{column}' + '_2sig_inlires'] = (df[[f'{column}']]<=df[[f'{column}']].quantile(.95)) & (df[[f'{column}']]>=df[[f'{column}']].quantile(.05))
        df[f'{column}' + '_3sig_inlires'] = (df[[f'{column}']]<=df[[f'{column}']].quantile(.997300203936740)) & (df[[f'{column}']]>=df[[f'{column}']].quantile(.002699796063259985))
    
    # loop through list and print a scatterplot showing outliers
    for column in lst:

        plt.figure(figsize=(16, 6))
        sns.scatterplot(data=df[f'{column}'][(df[f'{column}_2sig_inlires'] == True) & (df[f'{column}_3sig_inlires'] == True)])
        sns.scatterplot(data=df[f'{column}'][(df[f'{column}_2sig_inlires'] == False) & (df[f'{column}_3sig_inlires'] == False)])
        sns.scatterplot(data=df[f'{column}'][(df[f'{column}_2sig_inlires'] == False) & (df[f'{column}_3sig_inlires'] == True)])
        plt.title(f"{column} Outliers")
        plt.show()