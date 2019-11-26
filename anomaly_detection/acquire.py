import pandas as pd

def get_the_data():
    '''
    Acquires data from anonymized-curriculum-access.csv
    '''

    df = pd.read_csv('anonymized-curriculum-access.csv', sep=' ', header=None, engine='python')
    
    return df


