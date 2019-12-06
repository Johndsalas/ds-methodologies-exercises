from requests import get
from bs4 import BeautifulSoup
import os

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