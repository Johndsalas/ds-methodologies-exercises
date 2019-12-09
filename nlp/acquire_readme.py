from requests import get
from bs4 import BeautifulSoup
import os
import pandas as pd

import unicodedata
import re
import json

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords


def get_sub_list(topic_url):
    '''
    Takes in a topic url and returns a sub_list of dictionaries for that topic
    '''
    # make web request using selected url
    headers = {'User-Agent': 'Technomancer'}
    response = get(topic_url, headers=headers)

    # cerate soup object
    soup = BeautifulSoup(response.text,'html.parser')

    # get articles from soup object
    articles = soup.select(".form")
    print(articles)
    # create empty list
    sub_list = []
    
    # itterate through articles 
    for article in articles:
        
        # use soup object to identify title, content and catagory of article
        text = article.select("[form='action']")[0].get_text()

        # create dictionary with title, content, and catagory and append it to the sub_list
        sub_list.append({"Text":content})

    # return sub_list
    return sub_list

get_sub_list('https://github.com/trending')

# def get_READMEs():
#     '''
#     Function gathers a list of news articles based on a list of topic urls
#     creates a dictionary out of each artical seperating title, content, and catagory,
#     adds each dictionary to a list
#     combines each list to a master list
#     turns the list into a dataframe
#     writes the dataframe to a file
#     and returns the dataframe.
    
#     If file already exists function will simply return that file as a dataframe.
#     '''
    
#     # if file exists open it
#     if os.path.exists('READMEs.csv'):
#         df = pd.DataFrame(pd.read_csv('READMEs.csv'))
#         return df
    
#     else:
        
#         # create list of topic URLs
#         topic_urls = [ https://github.com/trending]
    
#         # create empty master list
#         master_list = []
    
#         # itterate through topic urls
#         for topic_url in topic_urls:
        
#             # call pass topic URL to get_sublist and append resulting sub_list to master list
#             # use .extend to flatten list
#             sub_list = get_sub_list(topic_url)
            
#             # append sub_list to master list
#             master_list.extend(sub_list)
        
#         # transform master list into a data frame
#         df = pd.DataFrame(master_list)
        
#         # write master list to data frame
#         df.to_csv('inshorts_news_articles.csv')
    
#         # return data frame
#         return df

