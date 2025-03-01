import warnings
warnings.filterwarnings('ignore')
import statsmodels.api as sm
import pandas as pd
import numpy as np
from scipy import stats
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from env import host, user, password


def get_telco():

    database = "telco_churn"

    def get_db_url(user,host,password,database):

        url = f'mysql+pymysql://{user}:{password}@{host}/{database}'
    
        return url

    url = get_db_url(user,host,password,database)

    query =  sql ='''
        select
        cust.`customer_id`,
        cust.`gender`,
        cust.`gender` = 'Male' is_male,
        cust.`senior_citizen`,
        cust.`partner` = 'Yes' partner,
        cust.`dependents` = 'Yes' dependents,
        cust.`partner` = 'Yes' or cust.`dependents` = 'Yes' family,
        2 * case when cust.`partner` = 'Yes' then 1 else 0 end + case when cust.`dependents` = 'Yes' then 1 else 0 end partner_deps_id,
        concat(
            case when cust.`partner` = 'Yes' then 'Has ' else 'No ' end,
            'partner, ',
            case when cust.`dependents` = 'Yes' then 'has ' else 'no ' end,
            'dependents') partner_deps,
        cust.`tenure`,
        cust.`phone_service` = 'Yes' phone_service,
        cust.`multiple_lines` = 'Yes' multiple_lines,
        case when cust.`phone_service` = 'Yes' then 1 else 0 end + case when cust.`multiple_lines` = 'Yes' then 1 else 0 end phone_service_id,
        case when cust.`phone_service` = 'Yes' then case when cust.`multiple_lines` = 'Yes' then 'Multiple Lines' else 'Single Line' end else 'No Phone' end phone_service_type,
        cust.`internet_service_type_id` <> 3 internet_service,
        cust.`internet_service_type_id` = 1 has_dsl,
        cust.`internet_service_type_id` = 2 has_fiber,
        cust.`online_security` = 'Yes' online_security,
        cust.`online_backup` = 'Yes' online_backup,
        cust.`online_security` = 'Yes' or cust.`online_backup` = 'Yes' online_security_backup,
        cust.`device_protection` = 'Yes' device_protection,
        cust.`tech_support` = 'Yes' tech_support,
        cust.`streaming_tv` = 'Yes' streaming_tv,
        cust.`streaming_movies` = 'Yes' streaming_movies,
        cust.`streaming_tv` = 'Yes' or `streaming_movies` = 'Yes' streaming_services,
        cust.`contract_type_id`,
        ct.`contract_type`,
        cust.`contract_type_id` = 1 on_contract,
        case when cust.`contract_type_id` = 1 then 1 else case when cust.`contract_type_id` = 2 then 12 else 24 end end contract_duration,
        cust.`paperless_billing` = 'Yes' paperless_billing,
        cust.`payment_type_id`,
        pt.`payment_type`,
        pt.`payment_type` like '%%auto%%' auto_pay,
        cust.`monthly_charges`,
        case when cust.`total_charges` = '' then 0 else cast(cust.`total_charges` as decimal(11,2)) end total_charges,
        case when cust.`churn` = 'Yes' then 1 else 0 end churn
    from 
        customers cust
    left join 
        contract_types ct
        using(`contract_type_id`)
    left join 
        internet_service_types ist
        using(`internet_service_type_id`)
    left join 
        payment_types pt
        using(`payment_type_id`)
        '''

    df = pd.read_sql(query, url)

    return df


def split_data_whole(df,train_pct=.7):

    train, test = train_test_split(df, train_size = train_pct, random_state = 999)

    return train, test
