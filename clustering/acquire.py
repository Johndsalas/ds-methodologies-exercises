import pandas as pd
import env

def get_connection(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

# Remove any properties that are likely to be something other than a single unit properties (e.g. no duplexes, no land/lot, ...). There are multiple ways to estimate that a property is a single unit, and there is not a single "right" answer.

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



def get_iris_data():
    query = '''
    SELECT petal_length, petal_width, sepal_length, sepal_width, species_id, species_name
    FROM measurements m
    JOIN species s USING(species_id)
    '''
    return pd.read_sql(query, get_connection('iris_db'))

def get_mallcustomer_data():
    df = pd.read_sql('SELECT * FROM customers;', get_connection('mall_customers'))
    return df.set_index('customer_id')