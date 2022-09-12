import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")
from env import user, password, host
import acquire
import prepare
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import os 

def prep_zillow(df):
    '''prep the zzillow dataset by renaming the columns and 
    creating two new columns name decade and totalrooms
    i used df.drop to drop all of the null in this dataset 
    converted fips and year as objects'''
    #renaming all the columns again
    df = df.rename(columns={'bedroomcnt': 'Bedrooms', 'bathroomcnt': 'Bathrooms','calculatedfinishedsquarefeet':'Squarefeet',    "taxvaluedollarcnt":'TaxesTotal','yearbuilt':'Year','fips':'Fips','regionidcounty':'County','regionidzip':'Zip','numberofstories':'Stories','parcelid':'Parcelid'})
    #changing these column into objects
    df.Fips = df.Fips.astype(object)
    df.Year = df.Year.astype(object)
    df.Stories = df.Stories.astype(object)
    #creating a column name total rooms by accounting bath and bed rooms together
    df['TotalRooms'] = df['Bathrooms']+df['Bedrooms']
    #was trying to drop the null or replace them
    df=df.replace('NaN','0')
    df = df.drop(columns=('Stories'))
    things = ['Year', 'TaxesTotal', 'Squarefeet']
    for col in things:
        q1,q3 = df[col].quantile([.25,.75])
        iqr = q3-q1
        upper = q3 + 1.5*iqr
        lower = q1 - 1.5*iqr

        df = df[(df[col] > lower) & (df[col] < upper)]
    df['Decade'] = pd.cut(df.Year, bins=[1800,1850,1900,1910,1920,1930,1940,1950,1960,1970,1980,1990,2000,2010,2020],labels=['1800', '1850', '1900', '1910', '1920', '1930', '1940', '1950', '1960', '1970','1980','1990','2000','2010'])
    df.Decade = df.Decade.astype(float)
    #created a column connecting years into decades plus drop nulls
    df=df.replace('','0')
    df = df.fillna(0)
    #making decade into a int
    df.Decade = df.Decade.astype(int)
    #location split area codes 
    df['location'] = df.Fips.map({6037: 'Los_Angeles', 6059: 'Orange', 6111:'Ventura'})
    dummies = pd.get_dummies(df.Fips)
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(columns='Fips')
    #re arrange the columns back into place
    df.columns=['Bedrooms','Bathrooms','Squarefeet','TaxesTotal','Year','County','Zip','latitude','longitude','TotalRooms','Decade','location' ,'los_angeles', 'orange', 'ventura']
    #converted into a int
    df.los_angeles = df.los_angeles.astype(int)
    df.orange = df.orange.astype(int)
    df.ventura = df.ventura.astype(int)
    return df
     

def remove_outliers(df, k, col_list):
    ''' remove outliers from a list of columns in a dataframe 
        and return that dataframe
    '''
    
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df

def my_train_test_split(df):
    '''
    Takes in a dataframe and target (as a string). Returns train, validate, and test subset 
    dataframes with the .2/.8 and .25/.75 splits to create a final .2/.2/.6 split between datasets
    '''
    train, test = train_test_split(df, test_size = .2, random_state=123)
    train, validate = train_test_split(train, test_size = .25, random_state=123)
    
    return train, validate, test



def prepare_zillow_train(df, target, col_list):
    # remove all outliers from dataset
    df = remove_outliers(df, 1.5, col_list)    
    # splitting data into train, validate, test
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)
    
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    
    columns_to_scale = col_list
    
    #     make the thing
    scaler = MinMaxScaler()
    #     fit the thing
    scaler.fit(train[columns_to_scale])
    # applying the scaler:
    train_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(train[columns_to_scale]),
                                                  columns=train[columns_to_scale].columns.values).set_index([train.index.values])
                                                  
    validate_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(validate[columns_to_scale]),
                                                  columns=validate[columns_to_scale].columns.values).set_index([validate.index.values])
    
    test_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(test[columns_to_scale]),
                                                 columns=test[columns_to_scale].columns.values).set_index([test.index.values])
    
    # split train into X (dataframe, drop target) & y (series, keep target only)
    X_train = train_scaled.drop(columns=[target])
    y_train = train_scaled[target]

    # split validate into X (dataframe, drop target) & y (series, keep target only)
    X_validate = validate_scaled.drop(columns=[target])
    y_validate = validate_scaled[target]

    # split test into X (dataframe, drop target) & y (series, keep target only)
    X_test = test_scaled.drop(columns=[target])
    y_test = test_scaled[target]
    
    return train, validate, test, X_train, y_train, X_validate, y_validate, X_test, y_test




def wrangle_zillow():

    train, validate, test, X_train, y_train, X_validate, y_validate, X_test, y_test = prepare_zillow_train(prepare.prep_zillow(acquire.sqlclean_zillow()))

    return train, validate, test, X_train, y_train, X_validate, y_validate, X_test, y_test

def impute_mode(train, validate, test, col):
    '''
    Takes in train, validate, and test as dfs, and column name (as string) and uses train 
    to identify the best value to replace nulls in embark_town
    
    Imputes the most_frequent value into all three sets and returns all three sets
    '''
    imputer = SimpleImputer(strategy='most_frequent')
    imputer = imputer.fit(train[[col]])
    train[[col]] = imputer.transform(train[[col]])
    validate[[col]] = imputer.transform(validate[[col]])
    test[[col]] = imputer.transform(test[[col]])
    
    return train, validate, test

def scale_data(train, val, test, cols_to_scale):
    train_scaled = train.copy()
    val_scaled = val.copy()
    test_scaled = test.copy()
    
    scaler = MinMaxScaler()
    scaler.fit(train[cols_to_scale])
    
    train_scaled[cols_to_scale] = pd.DataFrame(scaler.transform(train[cols_to_scale]),
                                               columns = train[cols_to_scale].columns.values).set_index([train.index.values])
    val_scaled[cols_to_scale] = pd.DataFrame(scaler.transform(val[cols_to_scale]),
                                               columns = val[cols_to_scale].columns.values).set_index([val.index.values])
    test_scaled[cols_to_scale] = pd.DataFrame(scaler.transform(test[cols_to_scale]),
                                               columns = test[cols_to_scale].columns.values).set_index([test.index.values])
    
    return train_scaled, val_scaled, test_scaled





def model_setup(train_scaled, train, val_scaled, val, test_scaled, test):

    # Set up X and y values for modeling
    X_train, y_train = train_scaled.drop(columns=['TaxesTotal','location','Decade']), train.TaxesTotal
    X_val, y_val = val_scaled.drop(columns=['TaxesTotal','location','Decade']), val.TaxesTotal
    X_test, y_test = test_scaled.drop(columns=['TaxesTotal','location','Decade']), test.TaxesTotal

    # make them a dataframes
    y_train = pd.DataFrame(y_train)
    y_val = pd.DataFrame(y_val)
    y_test = pd.DataFrame(y_test)
    
    return X_train, y_train, X_val, y_val, X_test, y_test


#acquire------------------------------------------------------------------------------------------------------------------------#--------------------------------------------------------------------------------------------------------------------------------#--------------------------------------------------------------------------------------------------------------------------------

#creating a connect function to connected to the code up servers
def get_connection(db, user=env.user, host=env.host, password=env.password):
    '''initiates sql connection'''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'
    
#csv clean a downloaded version of the zillow data
def clean_zillow():
    '''Read zillow csv file into a pandas DataFrame,
    renamed all of the columuns, replace NaN values with 0 ,
    keep all the 0 values, convert all columns to int64,
    return cleaned zillow DataFrame'''
    df=pd.read_csv('zillow.csv')
    df = df.rename(columns={'bedroomcnt': 'Bedrooms', 'bathroomcnt': 'Bathrooms','calculatedfinishedsquarefeet':'Squarefeet',
                       "taxvaluedollarcnt":'TaxesTotal','yearbuilt':'Year','taxamount':'Taxes','fips':'Fips'})

    return df
#csv clean a downloaded version of the zillow data
def clean_zillow2():
    '''Read zillow csv file into a pandas DataFrame,
    renamed all of the columuns, replace NaN values with 0 ,
    keep all the 0 values, convert all columns to int64,
    return cleaned zillow DataFrame'''
    df=pd.read_csv('zillow3.csv')
    df=df.drop(columns=['taxamount','taxdelinquencyflag','taxdelinquencyyear','censustractandblock','assessmentyear',
                 'buildingqualitytypeid','buildingclasstypeid','decktypeid','calculatedbathnbr','fireplaceflag',
                 'structuretaxvaluedollarcnt','landtaxvaluedollarcnt','basementsqft','architecturalstyletypeid',
                "unitcnt",'yardbuildingsqft17','yardbuildingsqft26','parcelid.1','typeconstructiontypeid',
                'threequarterbathnbr','finishedsquarefeet13','finishedsquarefeet12','finishedfloor1squarefeet',
                 'finishedsquarefeet6','finishedsquarefeet50','finishedsquarefeet15','regionidneighborhood',
                 'fullbathcnt','fireplacecnt','roomcnt','garagecarcnt','garagetotalsqft','regionidcity',
                'hashottuborspa','rawcensustractandblock','propertyzoningdesc','storytypeid','logerror','pooltypeid2',
                'pooltypeid7','pooltypeid10','propertycountylandusecode','poolsizesum','poolcnt','heatingorsystemtypeid',
                'airconditioningtypeid','lotsizesquarefeet','propertylandusetypeid','id','transactiondate','propertylandusedesc'])
    df = df.rename(columns={'bedroomcnt': 'Bedrooms', 'bathroomcnt': 'Bathrooms','calculatedfinishedsquarefeet':'Squarefeet',
                       "taxvaluedollarcnt":'TaxesTotal','yearbuilt':'Year','fips':'Fips','regionidcounty':'County','regionidzip':'Zip','numberofstories':'Stories'
                       ,'parcelid':'Parcelid'})
    df=df.replace('','0')
    
    return df
#pulling from the code up server the zillow data frame   
def sqlclean_zillow():
    query = """
    SELECT bedroomcnt,bathroomcnt,calculatedfinishedsquarefeet,taxvaluedollarcnt,yearbuilt
    ,fips,regionidcounty,regionidzip,numberofstories,latitude,longitude FROM properties_2017
    JOIN propertylandusetype
    USING(propertylandusetypeid)
    JOIN predictions_2017
    USING(parcelid)
    WHERE propertylandusedesc = 'Single Family Residential'"""

    url = f"mysql+pymysql://{env.user}:{env.password}@{env.host}/zillow"
    df = pd.read_sql(query,url)

    return df



#modeling------------------------------------------------------------------------------------------------------------------------#--------------------------------------------------------------------------------------------------------------------------------#--------------------------------------------------------------------------------------------------------------------------------


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










