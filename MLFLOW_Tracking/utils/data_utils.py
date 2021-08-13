##### Data_utils.py#####
# Version: 1.1
# @Author : Prayank kulshrestha
# References : https://github.com/dmatrix/mlflow-workshop-part-1

import pandas as pd
import numpy as np
import os,re
import matplotlib as plt
import seaborn as sns


class utils:
    
    @staticmethod
    def load_data(path):
        '''
        load the input training file
        :param path: input file path as string
        :return: pandas dataframe object
        '''

        df = pd.read_csv(path)

        return df
    
    @staticmethod
    def plot_graph(x_data, y_data, x_label, y_label, title):
        '''
        Plot graph based on given input data
        use matplotlib for plotting the graph
        :param x_data: data for x_axis
        :param x_data : data for y_axis
        :param x_label: label for x_axis
        :param y_label : lable for y_axis
        :param title: title for the graph
        :return: None
        '''
        plt.figure(figsize= (10,5))
        plt.plot(x_data,y_data)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()
    

    @staticmethod
    def data_check(df):
        '''
        Print the Quick data Quality check
        :param df: input pandas dataframe
        :return : None
        '''

        print(f"Data shape is {df.shape}")
        print(f"\nTotal duplicate values in data is {df.duplicated().sum()}")
        
        # gives some infos on columns types and numer of null values
        train_info=pd.DataFrame(df.dtypes).T.rename(index={0:'column type'})
        train_info=train_info.append(pd.DataFrame(df.isnull().sum()).T.rename(index={0:'null values (nb)'}))
        train_info=train_info.append(pd.DataFrame(df.isnull().sum()/df.shape[0]*100).T.
                                rename(index={0:'null values (%)'}))

        print("\n Data  Null values Description is")
        print(train_info.head())
    
    @staticmethod
    def get_mlflow_directory_path(*paths, create_dir=True):
        """
        Get the current running path where mlruns is created. This is the directory from which
        the python file containing MLflow code is executed. This method is used for artifacts, such
        as images, where we want to store plots.
        :param paths: list of directories below mlfruns, experimentID, mlflow_run_id
        :param create_dir: detfault is True
        :return: path to directory.
        """

        cwd = os.getcwd()
        dir_ = os.path.join(cwd, "mlruns", *paths)
        if create_dir:
            if not os.path.exists(dir_):
                os.mkdir(dir_, mode=0o755)
        return dir
    






