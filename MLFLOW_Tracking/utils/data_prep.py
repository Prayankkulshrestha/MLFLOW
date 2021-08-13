import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn import preprocessing
from sklearn import model_selection

from warnings import filterwarnings
filterwarnings('ignore')



class data_preparation:

    def __init__(self,df):
        '''
        Initilize the data Preparation with pandas dataframe object
        :param df: pandas dataframe object
        '''

        self.df = df
    

    def data_prep_(self):
        '''
        Data Preparation for modelling, This method perform the following task
        1. Fill the Null values
        2. LabelEncoding 
        3. StandardScaler
        4. Drop the not required columns
        5. return the clean data

        :return: pandas dataframe object
        '''
        self.df['TotalCharges'] = self.df['TotalCharges'].replace(' ',np.nan)
        self.df['TotalCharges'] = self.df['TotalCharges'].astype('float')
        self.df['TotalCharges'] = self.df['TotalCharges'].fillna(self.df['TotalCharges'].median())

        label_col = [col for col in self.df.columns if col not in 
                                    ['customerID','MonthlyCharges','TotalCharges','tenure']]

        for each_col in label_col:
            lbl = preprocessing.LabelEncoder()
            self.df.loc[:,each_col] = lbl.fit_transform(self.df[each_col])
        
        for each_col in ['MonthlyCharges','TotalCharges','tenure']:
            std = preprocessing.StandardScaler()
            self.df.loc[:,each_col] = std.fit_transform(self.df[each_col].values.reshape(-1,1))
        
        self.df.drop('customerID',axis=1,inplace=True)

        return self.df
    


