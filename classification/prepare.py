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

def prep_iris(df):

    df = df.drop(columns=['species_id','measurement_id'])
    df = df.rename(columns={'species_name':'species'})
    encoder = LabelEncoder()
    df.species.fillna('Unknown')
    encoder.fit(df.species)
    array = encoder.transform(df.species)
    names = list(encoder.inverse_transform(array))
    return names

def prep_titanic_data(tdf):
    
    tdf = tdf.fillna(np.nan)
    tdf = tdf.drop(columns='deck')
    tdf = tdf[pd.notna(tdf.age)]

    imp_mode = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    imp_mode.fit(tdf[['embarked']])
    tdf['embarked'] = imp_mode.transform(tdf[['embarked']])

    imp_mode = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    imp_mode.fit(tdf[['embark_town']])
    tdf['embark_town'] = imp_mode.transform(tdf[['embark_town']])

    int_encoder = LabelEncoder()
    int_encoder.fit(tdf.embarked)
    tdf_embarked = int_encoder.transform(tdf.embarked)

    scaler = MinMaxScaler()
    scaler.fit(tdf[['fare']])
    tdf.fare = scaler.transform(tdf[['fare']])

    scaler = MinMaxScaler()
    scaler.fit(tdf[['age']])
    tdf.age = scaler.transform(tdf[['age']])

    return tdf

