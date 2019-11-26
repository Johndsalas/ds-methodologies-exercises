import pandas as pd

def prepare_the_data(df):
    '''
    Prepares the data from anonymized-curriculum-access.csv
    '''
    
    # change column names
    df = df.rename(columns={0:'date', 1: 'time', 2: 'http', 3: 'person', 4: 'group', 5: 'ipaddress'})
    
    # create timestamp column
    df['timestamp'] = df['date'] + ' ' + df['time']
    
    # set timestamp to date time
    df.timestamp = pd.to_datetime(df.timestamp)
    
    # set timestamp to index
    df = df.set_index('timestamp')
    
    # drop extra columns
    df = df.drop(columns=['date','time'])
    
    # fill missing values with 0
    df = df.fillna(0)
    return df