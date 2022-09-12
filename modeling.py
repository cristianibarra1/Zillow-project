import pandas as pd
import numpy as np 
import acquire
import prepare
import pandas as pd
import numpy as np
import os
import scipy.stats as stats
from sklearn.impute import SimpleImputer
import sklearn.preprocessing
# Viz imports
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt

# Modeling imports
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
# for modeling and evaluation
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures     
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures
warnings.filterwarnings("ignore")
# Custom module imports
import acquire
import prepare
import modeling
α = .05
alpha= .05
#test----------------------------------------------------------------------------------------------------------------------------#--------------------------------------------------------------------------------------------------------------------------------#-------------------------------------------------------------------------------------------------------------------------------- 
def test1(train):
    ''' 
    This function takes in the train dataset and outputs the Chi-Square results for hypothesis 2b
    in the zillow regression project addressing the relationship between bedroom and bathroom counts.
    '''
    # Set alpha
    α = 0.05

    # Create observed data
    observed = pd.crosstab(train.Bedrooms, train.TaxesTotal)

    # Run chi-square test
    chi2,pval,degf,expected = stats.chi2_contingency(observed)

    # Evaluate results by comparing the p-value with alpha
    if pval < α:
        print('''Reject the Null Hypothesis.
    Findings suggest there is an association between bedrooms and taxestotal.''')
    else:
        print('''Fail to reject the Null Hypothesis.
    Findings suggest there is not an association between bedrooms and taxestotal.''')
        
####      
###bathroom test
def test2(train):
    ''' 
    This function takes in the train dataset and outputs the Chi-Square results for hypothesis 2b
    in the zillow regression project addressing the relationship between bath and taxestotal.
    '''
    # Set alpha
    α = 0.05

    # Create observed data
    observed = pd.crosstab(train.Bathrooms, train.TaxesTotal)

    # Run chi-square test
    chi2,pval,degf,expected = stats.chi2_contingency(observed)

    # Evaluate results by comparing the p-value with alpha
    if pval < α:
        print('''Reject the Null Hypothesis.
    Findings suggest there is an association between bathrooms and taxestoal.''')
    else:
        print('''Fail to reject the Null Hypothesis.
    Findings suggest there is not an association between bathrooms and taxs total.''')

        
##squarefeet test
def test3(train):
    ''' 
    This function takes in the train dataset and outputs the Chi-Square results for hypothesis 2b
    in the zillow regression project addressing the relationship between bedroom and bathroom counts.
    '''
    # Set alpha
    α = 0.05

    # Create observed data
    observed = pd.crosstab(train.Squarefeet, train.TaxesTotal)

    # Run chi-square test
    chi2,pval,degf,expected = stats.chi2_contingency(observed)

    # Evaluate results by comparing the p-value with alpha
    if pval < α:
        print('''Reject the Null Hypothesis.
     Findings suggest there is an association between Squarefeet and taxestoal.''')
    else:
        print('''Fail to reject the Null Hypothesis.
     Findings suggest there is not an association between Squarefeet and taxs total.''')
        
##decade test
def test4(train):
    ''' 
    This function takes in the train dataset and outputs the Chi-Square results for hypothesis 2b
    in the zillow regression project addressing the relationship between decade and taxes.
    '''
    # Set alpha
    α = 0.05

    # Create observed data
    observed = pd.crosstab(train.Decade, train.TaxesTotal)

    # Run chi-square test
    chi2,pval,degf,expected = stats.chi2_contingency(observed)

    # Evaluate results by comparing the p-value with alpha
    if pval < α:
        print('''Reject the Null Hypothesis.
    Findings suggest there is an association between Decade and taxestoal.''')
    else:
        print('''Fail to reject the Null Hypothesis.
    Findings suggest there is not an association between Decade and taxs total.''')
###county test      
def test5(train):
    ''' 
    This function takes in the train dataset and outputs the Chi-Square results for hypothesis 2b
    in the zillow regression project addressing the relationship between county and taxes.
    '''
    # Set alpha
    α = 0.05

    # Create observed data
    observed = pd.crosstab(train.County, train.TaxesTotal)

    # Run chi-square test
    chi2,pval,degf,expected = stats.chi2_contingency(observed)

    # Evaluate results by comparing the p-value with alpha
    if pval < α:
        print('''Reject the Null Hypothesis.
        Findings suggest there is an association between County and taxestoal.''')
    else:
        print('''Fail to reject the Null Hypothesis.
        Findings suggest there is not an association between County and taxs total.''')




#modeling------------------------------------------------------------------------------------------------------------------------#--------------------------------------------------------------------------------------------------------------------------------#--------------------------------------------------------------------------------------------------------------------------------   
    
def Modeling_function(X_train, y_train, X_val, y_val):
    ''' 
    This function takes in the X and y objects and then runs the following models:
    Using y_train mean to acquire baseline,
    LarsLasso Alpha=1,
    Quadratic Linear Regression
    
    Returns a DataFrame with the results.
    '''
    #most models are at 2 or 1
    # Baseline Model
    pred_mean = y_train.TaxesTotal.mean()
    y_train['pred_mean'] = pred_mean
    y_val['pred_mean'] = pred_mean
    rmse_train = mean_squared_error(y_train.TaxesTotal, y_train.pred_mean, squared=False)
    rmse_val = mean_squared_error(y_val.TaxesTotal, y_val.pred_mean, squared=False)

    # save the results
    metrics = pd.DataFrame(data=[{
        'Model': 'baseline_mean',
        'Train_rmse': rmse_train,
        'Train_r2': explained_variance_score(y_train.TaxesTotal, y_train.pred_mean),
        'Val_rmse': rmse_val,
        'Val_r2': explained_variance_score(y_val.TaxesTotal, y_val.pred_mean)}])

    # LassoLars Model
    # run the model
    lars = LassoLars(alpha=2)
    lars.fit(X_train, y_train.TaxesTotal)
    y_train['pred_lars'] = lars.predict(X_train)
    rmse_train = mean_squared_error(y_train.TaxesTotal, y_train.pred_lars, squared=False)
    y_val['pred_lars'] = lars.predict(X_val)
    rmse_val = mean_squared_error(y_val.TaxesTotal, y_val.pred_lars, squared=False)

    # save the results
    metrics = metrics.append({
        'Model': 'Lars_alpha(2)',
        'Train_rmse': rmse_train,
        'Train_r2': explained_variance_score(y_train.TaxesTotal, y_train.pred_lars),
        'Val_rmse': rmse_val,
        'Val_r2': explained_variance_score(y_val.TaxesTotal, y_val.pred_lars)}, ignore_index=True)

    # Polynomial Models
    # set up the model
    pf = PolynomialFeatures(degree=1)
    X_train_d2 = pf.fit_transform(X_train)
    X_val_d2 = pf.transform(X_val)
    
    # run the model
    lm2 = LinearRegression()
    lm2.fit(X_train_d2, y_train.TaxesTotal)
    y_train['pred_lm2'] = lm2.predict(X_train_d2)
    rmse_train = mean_squared_error(y_train.TaxesTotal, y_train.pred_lm2, squared=False)
    y_val['pred_lm2'] = lm2.predict(X_val_d2)
    rmse_val = mean_squared_error(y_val.TaxesTotal, y_val.pred_lm2, squared=False)

    # save the results
    metrics = metrics.append({
        'Model': 'Depth(1)',
        'Train_rmse': rmse_train,
        'Train_r2': explained_variance_score(y_train.TaxesTotal, y_train.pred_lm2),
        'Val_rmse': rmse_val,
        'Val_r2': explained_variance_score(y_val.TaxesTotal, y_val.pred_lm2)}, ignore_index=True)

    # set up the model
    pf = PolynomialFeatures(degree=2)
    X_train_d2 = pf.fit_transform(X_train)
    X_val_d2 = pf.transform(X_val)

    # run the model
    lm2 = LinearRegression()
    lm2.fit(X_train_d2, y_train.TaxesTotal)
    y_train['pred_lm2'] = lm2.predict(X_train_d2)
    rmse_train = mean_squared_error(y_train.TaxesTotal, y_train.pred_lm2, squared=False)
    y_val['pred_lm2'] = lm2.predict(X_val_d2)
    rmse_val = mean_squared_error(y_val.TaxesTotal, y_val.pred_lm2, squared=False)

    # save the results
    metrics = metrics.append({
        'Model': 'Depth(2)',
        'Train_rmse': rmse_train,
        'Train_r2': explained_variance_score(y_train.TaxesTotal, y_train.pred_lm2),
        'Val_rmse': rmse_val,
        'Val_r2': explained_variance_score(y_val.TaxesTotal, y_val.pred_lm2)}, ignore_index=True)

    return metrics


def modeling_best(X_train, y_train, X_val, y_val, X_test, y_test):
    # set up the model
    pf = PolynomialFeatures(degree=2)
    X_train_d2 = pf.fit_transform(X_train)
    X_val_d2 = pf.transform(X_val)
    X_test_d2 = pf.transform(X_test)

    # run the model
    lm2 = LinearRegression()
    lm2.fit(X_train_d2, y_train.TaxesTotal)
    y_train['pred_lm2'] = lm2.predict(X_train_d2)
    rmse_train = mean_squared_error(y_train.TaxesTotal, y_train.pred_lm2, squared=False)
    y_val['pred_lm2'] = lm2.predict(X_val_d2)
    rmse_val = mean_squared_error(y_val.TaxesTotal, y_val.pred_lm2, squared=False)
    y_test['pred_lm2'] = lm2.predict(X_test_d2)
    rmse_test = mean_squared_error(y_test.TaxesTotal, y_test.pred_lm2, squared=False)
    # save the results
    results = pd.DataFrame({'test':{'Test_rmse': rmse_test,'Test_r2': explained_variance_score(y_test.TaxesTotal,y_test.pred_lm2)}})
    results.dropna()
    
    return results


#graphs--------------------------------------------------------------------------------------------------------------------------#--------------------------------------------------------------------------------------------------------------------------------#--------------------------------------------------------------------------------------------------------------------------------
def plot_variable_pairs(train):
    cats = ['Bedrooms', 'Bathrooms',  'Year',  'Fips']
    nums = ['Squarefeet', 'Taxes', 'TaxesTotal','Decade']
    
    # make correlation plot
    train = train.drop(columns=['Fips']).corr()
    plt.figure(figsize=(12,8))
    sns.pairplot(train[nums].sample(1000), corner=True, kind='reg',plot_kws={'line_kws':{'color':'red'}})
    plt.show()
    
def bedrooms_price1(train):
     # make correlation plot
    plt.figure(figsize=(5,10))
    sns.set_theme(style="darkgrid")
    sns.jointplot(x="Bedrooms",y='TaxesTotal', data=train,
                  kind="reg", truncate=False,
                  color="m", height=7)
    #arrow to show path
    plt.arrow(1.9,.4,3,3, head_width=.2, color='red')
    plt.title('Bedrooms')
    plt.show()



def bedrooms_price(train):
    #plotting up graph comparing bedrooms and taxes while splitting up by locations
    brush = alt.selection(type='interval')
    alt.data_transformers.disable_max_rows()
    points = alt.Chart(train).mark_point().encode(
    x='Bedrooms',
    y='TotalRooms',
    color=alt.condition(brush, 'TaxesTotal', alt.value('lightgray'))).add_selection(brush)
    bars = alt.Chart(train).mark_bar().encode(
    y='TaxesTotal',
    color='location',
    x='Bedrooms').transform_filter(brush)
    
    return points & bars 

def squarefeet_taxes(train):
    # making aplot comparing squarefeet with taxestotal 
    plt.figure(figsize=(8,10))
    sns.set_theme(style="darkgrid")
    sns.jointplot(x="Squarefeet",y='TaxesTotal', data=train,
                  kind="reg", truncate=False,
                  color="m")
    plt.title('Cost per Squarefeet?')
    plt.show()
    
def squarefeet_taxes1(train):
    # making aplot comparing squarefeet with taxestotal 
    plt.figure(figsize=(8,10))
    sns.histplot(data=train, x='TaxesTotal')
    plt
    plt.title('Home cost')
    med = train[train.Squarefeet > train.Squarefeet.median()].TaxesTotal.median()
    plt.axvline(x=med, color='orange')
    plt.show()
    
    
def county_tax(train):
    #sspliting up location and comparing there taxes which one is the highest??
    plt.figure(figsize=(8,8))
    sns.histplot(data=train, x='TaxesTotal', hue='location', bins= 75)
    med = train[train.Squarefeet > train.Squarefeet.median()].TaxesTotal.median()
    plt.axvline(x=med, color='orange')
    plt.show()
    
def county_tax1(train):
    #sspliting up location and comparing there taxes which one is the highest??
    plt.figure(figsize=(5,10))
    sns.set_theme(style="darkgrid")
    sns.jointplot(x="TaxesTotal",y='County', data=train,hue='location',
                  color="m", height=7)
    plt.title('County taxes values?')
    plt.show()
    
def Bathrooms_tax(train):
    #bathrooms plot comparing bathrooms to taxestotal does it work ?
    brush = alt.selection(type='interval')
    alt.data_transformers.disable_max_rows()
    points = alt.Chart(train).mark_point().encode(
    x='Bathrooms',
    y='TotalRooms',
    color=alt.condition(brush, 'TaxesTotal', alt.value('lightgray'))).add_selection(brush)
    bars = alt.Chart(train).mark_bar().encode(
    y='TaxesTotal',
    color='location',
    x='Bathrooms').transform_filter(brush)
    
    return points & bars 

def Bathrooms_tax1(train):
    #bathrooms plot comparing bathrooms to taxestotal does it work ?
    plt.figure(figsize=(5,10))
    sns.set_theme(style="darkgrid")
    sns.jointplot(x="Bathrooms",y='TaxesTotal', data=train,
                  kind="reg", truncate=False,
                  color="m", height=7)
    plt.arrow(1.9,.4,3,3, head_width=.2, color='red')
    plt.title('Bathrooms')
    plt.show()

def decade_tax1(train):
    #Decadescompared to taxes and does it effect it ?
    plt.figure(figsize=(5,10))
    sns.set_theme(style="darkgrid")
    sns.jointplot(x="Decade",y='TaxesTotal', data=train,
                  kind="reg", truncate=False,xlim=(1750,2020),
                  color="m", height=7)
    plt.title('Decades to taxes')
    plt.show()
    
def decade_tax2(train):
    #Decadescompared to taxes and does it effect it ? while spliting by location
    plt.figure(figsize=(15,8))
    sns.barplot(data=train, x="Decade", y="TaxesTotal", hue="location")
    plt.title('Decade cost')
    plt.show()
