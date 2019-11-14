import requests
import pandas as pd
import sys

base_url = 'https://python.zach.lol'

# look at api for url
#print(requests.get(base_url).text) # /api/v1 /documintation

# store base_url + url found in API in response
response = requests.get('https://python.zach.lol/api/v1/sales')

# store responce.json() in data and use to view dictionary keys
data = response.json()
#data.keys()

# view data for desiered dictionary (payload) found 'items' in sub dictionary
#data['payload'].keys()

# create data frame out of items dictionary through payload dictionary
df = pd.DataFrame(data['payload']['sales'])
#df.head()

# print max-page and next-page max 
# page = 2 next page = 183

print('max_page: %s' % data['payload']['max_page'])
print('next_page: %s' % data['payload']['next_page'])

# adding additional pages

# itterate through additional pages of data and add each one to the data frame

start_page = 3
end_page = 183

while start_page != end_page + 1:

        print(f"on page {start_page}")

        # set responce to next page
        response = requests.get(base_url + data['payload']['next_page'])

        # set responce.json to data
        data = response.json()

        # concat data from page onto new data frame
        df = pd.concat([df, pd.DataFrame(data['payload']['sales'])])

        # add 1 to value of next page
        start_page += 1

df.shape