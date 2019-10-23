import pandas as pd
import numpy as np
import acquire as a
# import warnings
# warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from env import host, user, password

def wrangle_telco():

    database = "telco_churn"

    def get_db_url(user,host,password,database):

        url = f'mysql+pymysql://{user}:{password}@{host}/{database}'
    
        return url

    url = get_db_url(user,host,password,database)

    query = """ 
            SELECT customer_id, tenure, monthly_charges, total_charges 
            FROM customers 
            WHERE contract_type_id = 3          
            """

    df = pd.read_sql(query, url)

    df.total_charges = df.total_charges.str.strip().replace('', np.nan).astype(float)

    df = df.dropna()
   
    return df

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

from env import host, user, password


def get_sql(database,query):

    import pandas as pd
    import env

    def get_connection(db, user=env.user, host=env.host, password=env.password):
        return f'mysql+pymysql://{user}:{password}@{host}/{db}'

    df = pd.read_sql(query, get_connection(database))

    return df



def get_sql(database,query):

    import pandas as pd
    import env

    def get_connection(db, user=env.user, host=env.host, password=env.password):
        return f'mysql+pymysql://{user}:{password}@{host}/{db}'

    df = pd.read_sql(query, get_connection(database))

    return df

def get_titanic_data():

    import pandas as pd
    import env

    def get_connection(db, user=env.user, host=env.host, password=env.password):
        return f'mysql+pymysql://{user}:{password}@{host}/{db}'

    df = pd.read_sql('SELECT * FROM passengers', get_connection('titanic_db'))

    return df

def get_iris_data():

    import pandas as pd
    import env

    def get_connection(db, user=env.user, host=env.host, password=env.password):
        return f'mysql+pymysql://{user}:{password}@{host}/{db}'

    query = '''
            SELECT * 
            FROM measurements
            JOIN species
            USING (species_id)
            '''

    df = pd.read_sql(query, get_connection('iris_db'))

    return df


