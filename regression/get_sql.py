
# takes in a database and table name returns select * from table
def get_sql(database,table):

    database = database
    import pandas as pd
    from env import host, user, password

    def get_db_url(user,host,password,database):

        url = f'mysql+pymysql://{user}:{password}@{host}/{database}'
    
        return url

    url = get_db_url(user,host,password,database)

    query = f""" 
            SELECT * 
            FROM {table}        
            """

    df = pd.read_sql(query, url)

    return df

