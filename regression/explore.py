# Our scenario continues:

# As a customer analyst, I want to know who has spent the most money with us over their lifetime. 
# I have monthly charges and tenure, so I think I will be able to use those two attributes as features to estimate total_charges. 
# I need to do this within an average of $5.00 per customer.

# Create a file, explore.py, that contains the following functions for exploring your variables (features & target).
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

import env
import wrangle
import split_scale


df = wrangle.wrangle_telco()

train, test = split_scale.split_data_whole(df)

dataframe = train

# # Write a function, plot_variable_pairs(dataframe) that plots all of the pairwise relationships along with the regression line for each pair.

def plot_variable_pairs(dataframe):

    j = sns.scatterplot("tenure", "monthly_charges", data=dataframe,)
    plt.show()

plot_variable_pairs(dataframe)
# Write a function, months_to_years(tenure_months, df) that returns your dataframe with a new feature tenure_years, 
# in complete years as a customer.

# Write a function, plot_categorical_and_continous_vars(categorical_var, continuous_var, df), 
# that outputs 3 different plots for plotting a categorical variable with a continuous variable, 
# e.g. tenure_years with total_charges. 
# For ideas on effective ways to visualize categorical with continuous: https://datavizcatalogue.com/. 
# You can then look into seaborn and matplotlib documentation for ways to create plots.