import pandas as pd
import numpy as np
import os
import scipy.stats as stats

# Viz imports
import matplotlib.pyplot as plt
import seaborn as sns

# Modeling imports
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings("ignore")
# Custom module imports
import acquire
import prepare
import model
α = .05
alpha= .05


def plot_variable_pairs(df):
    cats = ['Bedrooms', 'Bathrooms',  'Year',  'Fips']
    nums = ['Squarefeet', 'Taxes', 'TaxesTotal','Decade']
    
    # make correlation plot
    df = df.drop(columns=['Fips']).corr()
    plt.figure(figsize=(12,8))
    sns.pairplot(train[nums].sample(1000), corner=True, kind='reg',plot_kws={'line_kws':{'color':'red'}})
    plt.show()