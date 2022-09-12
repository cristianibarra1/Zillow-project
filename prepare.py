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

##preparing data
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