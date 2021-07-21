from pandas import read_csv, concat, DataFrame

import pandas as pd

import os
import csv

def csv_to_pandas(fileName, chunksize):
    with open(fileName) as f:
        firstLine = f.readline()
    nbComma = firstLine.count(',')
    nbSemicolon = firstLine.count(';')
    if nbComma>nbSemicolon:
        dataset = pd.read_csv(fileName, chunksize=chunksize)
    else:
        dataset = pd.read_csv(fileName, sep =";", chunksize=chunksize)
    return dataset

class Dataset:
    def __init__(self):
        self.dataset_is_ready = False
        self.config = None
        self.nb_possibility = None
        self.info_goal = {}
        self.list_info_status = []

    def set(self, config, data):
        if data is not None:
            self.config = config
            self.data = data
            self.dataset_is_ready = True
            self.config['list_info_status'] = self.list_info_status
            self.config['nb_possibility'] = self.nb_possibility
            return self.config
        else:
            self.config = config

    def clean_data(self, data_in):
        """suppression des colonnes inutiles
        """
        data = data_in.copy()
        data.columns = self.config['list_format_column']
        list_format_column = self.config['list_format_column'][:]

        columns_to_del = [i for i,x in enumerate(list_format_column) if x=="disable"]
        data.drop(data.columns[columns_to_del], axis=1, inplace=True)

        for i in range(list_format_column.count('disable')):
            list_format_column.remove("disable")
        
        # renome les colonnes en fonction de leur type
        data.columns = list_format_column
        
        # tri par date
        if list_format_column.count('timestamp') == 1:
            data.sort_values(by='timestamp')
            data = data.reset_index(drop=True)
        
        return data


    def set_data(self, data):
        list_column_name = list(data.columns.values)
        mode = self.config['mode']

        # analyse des colonnes qui serviront à faire du one hot encoding
        rank = 0
        for (column_name, column_format) in zip(list_column_name, self.config['list_format_column']):
            if str(column_format) == 'status':
                info = {'name':column_name, 'rank':rank}
                self.list_info_status.append(info)
                rank +=1

        # enlèvement des colonnes 'disable' et tri par date si présence d'une colonne 'timestamp'
        data_cleaned = self.clean_data(data.copy())

        quality_column_goal = None
        list_quality_column = []
        list_quality_column_num = []
        for i, name in enumerate(data_cleaned.columns.tolist()):
            if str(name) == 'status':
                list_quality_column.append(data_cleaned.iloc[:, i])
                list_quality_column_num.append(i)

        list_data_one_hot = []

        for i, quality_column in enumerate(list_quality_column):
            list_data_one_hot.append(pd.get_dummies(quality_column))
            self.list_info_status[i]['possibility'] = list_data_one_hot[i].columns.values.tolist()

        if quality_column_goal is not None:
            goal_one_hot = pd.get_dummies(quality_column_goal)
            self.info_goal['possibility'] = goal_one_hot.columns.values.tolist()

        data_cleaned = data_cleaned.drop(data_cleaned.columns[list_quality_column_num], axis = 1)

        for i, data_one_hot in enumerate(list_data_one_hot):
            name = 'binaire_'+str(i)
            data_one_hot.columns = [name]*len(data_one_hot.columns)
            data_cleaned = data_cleaned.join(data_one_hot)

        if quality_column_goal is not None:
            name = 'goal'
            goal_one_hot.columns = [name]*len(goal_one_hot.columns)
            data_cleaned = data_cleaned.join(goal_one_hot)

        del list_data_one_hot
        print(f'DATASET: {type(data_cleaned)}')
        return data_cleaned

    def create_dataset(self, data):
        data_cleaned = self.set_data(data.copy())

