import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import env
import acquire
import prepare


def get_connection(db, user=env.user, host=env.host, password=env.password):
    '''initiates sql connection'''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'
    
#csv clean 
def clean_zillow():
    '''Read zillow csv file into a pandas DataFrame,
    renamed all of the columuns, replace NaN values with 0 ,
    keep all the 0 values, convert all columns to int64,
    return cleaned zillow DataFrame'''
    df=pd.read_csv('zillow.csv')
    df = df.rename(columns={'bedroomcnt': 'Bedrooms', 'bathroomcnt': 'Bathrooms','calculatedfinishedsquarefeet':'Squarefeet',
                       "taxvaluedollarcnt":'TaxesTotal','yearbuilt':'Year','taxamount':'Taxes','fips':'Fips'})

    return df

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
#sql clean   
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

