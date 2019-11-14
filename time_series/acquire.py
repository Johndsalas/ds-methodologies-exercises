import requests
import pandas as pd
import sys


# assign ulr to base url
base_url = 'https://python.zach.lol'

# look at api for url
print(requests.get(base_url).text) # /api/v1 /documintation

# store base_url + url found in API in response
response = requests.get('https://python.zach.lol/api/v1/items')

# store responce.json() in data and use to view dictionary keys
data = response.json()
data.keys()

# view data for desiered dictionary (payload) found 'items' in sub dictionary
data['payload'].keys()

# create data frame out of items dictionary through payload dictionary
df = pd.DataFrame(data['payload']['items'])
df.head()

# print max-page and next-page max 
# page = 3 next page = 2

print('max_page: %s' % data['payload']['max_page'])
print('next_page: %s' % data['payload']['next_page'])

# getting data from second page

# setting responce to next page (2)
response = requests.get(base_url + data['payload']['next_page'])

# setting new responce.jason() to data
data = response.json()

# add page 2 data onto data frame
df = pd.concat([df, pd.DataFrame(data['payload']['items'])])

# setting responce to next page (2)
response = requests.get(base_url + data['payload']['next_page'])

# setting new responce.jason() to data
data = response.json()

# add page 2 data onto data frame
df = pd.concat([df, pd.DataFrame(data['payload']['items'])])






# assign ulr to base url
base_url = 'https://python.zach.lol'

# look at api for url
print(requests.get(base_url).text) # /api/v1 /documintation

# view discription
response = requests.get(base_url + '/documentation')
#print(response.json()['payload'])

# store base_url + url found in API in response
response = requests.get('https://python.zach.lol/api/v1/sales')

# store responce.json() in data and use to view dictionary keys
data = response.json()
#data.keys()

# view store dictionary

# data['payload']['sales']

# create data frame out of items dictionary through payload dictionary
df = pd.DataFrame(data['payload']['sales'])
# df.head()

# print max-page and next-page max 
# page = 3 next page = 2

print('max_page: %s' % data['payload']['max_page'])
print('next_page: %s' % data['payload']['next_page'])















# no additional pages data frame complete 

# # assign ulr to base url
# base_url = 'https://python.zach.lol'

# # look at api for url
# #print(requests.get(base_url).text) # /api/v1 /documintation

# # store base_url + url found in API in response
# response = requests.get('https://python.zach.lol/api/v1/sales')

# # store responce.json() in data and use to view dictionary keys
# data = response.json()
# #data.keys()

# # view data for desiered dictionary (payload) found 'items' in sub dictionary
# #data['payload'].keys()

# # create data frame out of items dictionary through payload dictionary
# df = pd.DataFrame(data['payload']['sales'])
# #df.head()

# # print max-page and next-page max 
# # page = 2 next page = 183

# print('max_page: %s' % data['payload']['max_page'])
# print('next_page: %s' % data['payload']['next_page'])

# # adding additional pages

# # itterate through additional pages of data and add each one to the data frame

# next_page = 3
# max_page = 183

# while next_page =! end_page + 1:

#         print(f"on page {next_page}")

#         # set responce to next page
#         response = requests.get(base_url + data['payload']['next_page'])

#         # set responce.json to data
#         data = response.json()

#         # concat data from page onto new data frame
#         df = pd.concat([df, pd.DataFrame(data['payload']['items'])]).reset_index()

#         # add 1 to value of next page
#         next_page += 1

# df.shape

# def get_additional_pages(df,next_page,end_page):
# '''
# Takes in a data frame made from first page using requests, the next page number of the data, 
# and the last page number of the data. Returns data frame with all pages appended to it.
# '''
  



# add page 2 data onto data frame
df = pd.concat([df, pd.DataFrame(data['payload']['items'])]).reset_index()

# setting responce to next page (2)
response = requests.get(base_url + data['payload']['next_page'])

# setting new responce.jason() to data
data = response.json()

# add page 2 data onto data frame
df = pd.concat([df, pd.DataFrame(data['payload']['items'])]).reset_index()










def get_item_df():

    response = requests.get('https://python.zach.lol/api/v1/items')
    data = response.json()
    df = pd.DataFrame(data['payload']['items'])

    response = requests.get(base_url + data['payload']['next_page'])
    data = response.json()

    df = pd.concat([df, pd.DataFrame(data['payload']['items'])])

    response = requests.get(base_url + data['payload']['next_page'])
    data = response.json()

    df = pd.concat([df, pd.DataFrame(data['payload']['items'])])

    return df

get_item_df()


def get_stores_df():

    base_url = 'https://python.zach.lol'

    response = requests.get('https://python.zach.lol/api/v1/stores')

    data = response.json()

    df = pd.DataFrame(data['payload']['stores'])

    return df

get_stores_df()


# appending lesson answers to end of lesson

from os import path

import requests
import pandas as pd

BASE_URL = 'https://python.zach.lol'
API_BASE = BASE_URL + '/api/v1'

def get_store_data_from_api():
    url = API_BASE + '/stores'
    response = requests.get(url)
    data = response.json()
    return pd.DataFrame(data['payload']['stores'])

def get_item_data_from_api():
    url = API_BASE + '/items'
    response = requests.get(url)
    data = response.json()

    stores = data['payload']['items']

    while data['payload']['next_page'] is not None:
        print('Fetching page {} of {}'.format(data['payload']['page'] + 1, data['payload']['max_page']))
        url = BASE_URL + data['payload']['next_page']
        response = requests.get(url)
        data = response.json()
        stores += data['payload']['items']

    return pd.DataFrame(stores)

def get_sale_data_from_api():
    url = API_BASE + '/sales'
    response = requests.get(url)
    data = response.json()

    stores = data['payload']['sales']

    while data['payload']['next_page'] is not None:
        print('Fetching page {} of {}'.format(data['payload']['page'] + 1, data['payload']['max_page']))
        url = BASE_URL + data['payload']['next_page']
        response = requests.get(url)
        data = response.json()
        stores += data['payload']['sales']

    return pd.DataFrame(stores)

def get_store_data(use_cache=True):
    if use_cache and path.exists('stores.csv'):
        return pd.read_csv('stores.csv')
    df = get_store_data_from_api()
    df.to_csv('stores.csv', index=False)
    return df

def get_item_data(use_cache=True):
    if use_cache and path.exists('items.csv'):
        return pd.read_csv('items.csv')
    df = get_item_data_from_api()
    df.to_csv('items.csv', index=False)
    return df

def get_sale_data(use_cache=True):
    if use_cache and path.exists('sales.csv'):
        return pd.read_csv('sales.csv')
    df = get_sale_data_from_api()
    df.to_csv('sales.csv', index=False)
    return df

def get_opsd_data(use_cache=True):
    if use_cache and path.exists('opsd.csv'):
        return pd.read_csv('opsd.csv')
    df = pd.read_csv('https://raw.githubusercontent.com/jenfly/opsd/master/opsd_germany_daily.csv')
    df.to_csv('opsd.csv', index=False)
    return df