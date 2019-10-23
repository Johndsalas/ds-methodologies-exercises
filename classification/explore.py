import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import acquire as a
import prepare as p

df = a.get_titanic_data()
df = p.prep_titanic_data(df)

def loopy_graphs(df, target):
    features = list(df.columns[(df.dtypes == object) | (df.nunique()<5)])
    
    
    pop_rate = df[target].mean()

    for i, feature in enumerate(features):
        sns.barplot(feature,target,data=df,alpha=.6)
        plt.show()

def plot_violin(features, target, df):
    for descrete in df[features].select_dtypes([object,int]).columns.tolist():
        if df[descrete].nunique() <= 5:
            for continous in df[features].select_dtypes(float).columns.tolist():
                sns.violinplot(descrete, continous, hue=target,
                data=df, split=True, palette=['blue','orange'])
                plt.title(continous + 'x' + descrete)
                plt.ylabel(continous)
                plt.show()

