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

def get_blog_articles():
    '''
    function creates a list of dictionaries from codeup websites seperating each artical by title and body of the artical
    '''
    
    # if we already have the data, read it locally
    if os.path.exists('article.txt'):
        with open('article.txt') as f:
            return f.read()

    # if the data does not exist create it
    else: 
        
        # populate url list and create an empty dictionary
        url_lst= ['https://codeup.com/codeups-data-science-career-accelerator-is-here/','https://codeup.com/data-science-myths/','https://codeup.com/data-science-vs-data-analytics-whats-the-difference/','https://codeup.com/10-tips-to-crush-it-at-the-sa-tech-job-fair/','https://codeup.com/competitor-bootcamps-are-closing-is-the-model-in-danger/']
    
        article_dictionary = []
        
        # itterate through url list
        for url in url_lst:
    
            # make web request
            headers = {'User-Agent': 'Technomancer'}
            response = get(url, headers=headers)
            
            # parse data using beautiful soup
            soup = BeautifulSoup(response.text,features="lxml")
            
            # seperate artical into title and body 
            title = soup.select("title")
            article = soup.find('div', class_='mk-single-content')
            
            # create a mini-dictionary for title and body
            mini_dict = {"title":title[0].text,"article":article.text}
            
            # append mini-dictionary to main dictionary
            article_dictionary.append(mini_dict)
        
        # write main dictionary to text file
        with open('article.txt', 'w') as f:
            f.write(str(article_dictionary))
            
        # read file
        with open('article.txt', 'r') as f:
            return f.read()



def get_news_articles():
    '''
    Function gathers a list of news articles based on a list of topic urls
    creates a dictionary out of each artical seperating title, content, and catagory,
    adds each dictionary to a list
    combines each list to a master list
    turns the list into a dataframe
    writes the dataframe to a file
    and returns the dataframe.
    
    If file already exists function will simply return that file as a dataframe.
    '''
    
    # if file exists open it
    if os.path.exists('inshorts_news_articles.csv'):
        df = pd.DataFrame(pd.read_csv('inshorts_news_articles.csv'))
        return df
    
    else:
        
        # create list of topic URLs
        topic_urls = [
                "https://inshorts.com/en/read/business",
                "https://inshorts.com/en/read/sports",
                "https://inshorts.com/en/read/technology",
                "https://inshorts.com/en/read/entertainment"
                ]
    
        # create empty master list
        master_list = []
    
        # itterate through topic urls
        for topic_url in topic_urls:
        
            # call pass topic URL to get_sublist and append resulting sub_list to master list
            # use .extend to flatten list
            sub_list = get_sub_list(topic_url)
            
            # append sub_list to master list
            master_list.extend(sub_list)
        
        # transform master list into a data frame
        df = pd.DataFrame(master_list)
        
        # write master list to data frame
        df.to_csv('inshorts_news_articles.csv')
    
        # return data frame
        return df


def get_sub_list(topic_url):
    '''
    Takes in a topic url and returns a sub_list of dictionaries for that topic
    '''
    # make web request using selected url
    headers = {'User-Agent': 'Technomancer'}
    response = get(topic_url, headers=headers)

    # cerate soup object
    soup = BeautifulSoup(response.text)

    # get articles from soup object
    articles = soup.select(".news-card")
    
    # create empty list
    sub_list = []
    
    # itterate through articles 
    for article in articles:
        
        # use soup object to identify title, content and catagory of article
        title = article.select("[itemprop='headline']")[0].get_text()
        content = article.select("[itemprop='articleBody']")[0].get_text()
        category = response.url.split("/")[-1]

        # create dictionary with title, content, and catagory and append it to the sub_list
        sub_list.append({"title":title,"content":content,"category":category})

    # return sub_list
    return sub_list
