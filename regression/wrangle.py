# Throughout the exercises for Regression in Python lessons, you will use the following example scenario: 
# As a customer analyst, I want to know who has spent the most money with us over their lifetime. 
# I have monthly charges and tenure, so I think I will be able to use those two attributes as features to estimate total_charges. 
# I need to do this within an average of $5.00 per customer.
# The first step will be to acquire and prep the data. 
# Do your work for this exercise in a file named wrangle.py.
# Acquire customer_id, monthly_charges, tenure, and total_charges from telco_churn database for all customers with a 2 year contract.
# Walk through the steps above using your new dataframe. 
# You may handle the missing values however you feel is appropriate.
# End with a python file wrangle.py that contains the function, 
# wrangle_telco(), that will acquire the data and return a dataframe cleaned with no missing values.

import warnings
warnings.filterwarnings('ignore')

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

    df.replace(' ', np.nan, regex=True, inplace=True)
   
    df.total_charges = df.total_charges.dropna().astype(float)
   
    return df



