import progressbar
from pandas import read_csv, concat, DataFrame
#from scaler import Scaler

import pandas as pd
import numpy as np

from glob import glob
import json
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
        """self.list_train_X = None
        self.list_train_y = None"""
        self.config = None
        self.nb_possibility = None
        self.info_goal = {}
        self.list_info_status = []

    def set(self, config, data):
        if data is not None:
            self.config = config
            #self.create_dataset(data)
            self.data = data
            self.dataset_is_ready = True
            self.config['list_info_status'] = self.list_info_status
            #self.config['nb_entries'] = self.list_train_X.shape[2]
            self.config['nb_possibility'] = self.nb_possibility
            return self.config
        else:
            self.config = config
            """self.scaler_in = Scaler(load = self.config['scaler_in_save'])
            self.scaler_out = Scaler(load = self.config['scaler_out_save'])"""

    def generate_batches(self, file_name):
        lookback = self.config['lookback']
        lookahead = self.config['lookahead']

        # data_cleaned = self.set_data(data)

        c_size = (lookback+lookahead) * 256
        for gm_chunk in pd.read_csv('test_gerate_batches.csv', sep =";", chunksize=c_size):
            gm_chunk.columns = self.config['list_format_column']
            data_cleaned = self.clean_data(gm_chunk)
            data_formated = self.raw_data_to_data_formated(data_cleaned)
            x, y = self.dataToDataset(data_formated, lookahead)
            yield (x, y)

    def raw_data_to_data_formated(self, raw_data):
        mode = self.config['mode']

        if raw_data.columns.tolist().count('timestamp') != 0:
            del raw_data['timestamp']

        # trouver la colonne qui correspond au Goal
        quality_column_goal = None
        list_quality_column = []
        list_quality_column_num = []
        for i, name in enumerate(raw_data.columns.tolist()):
            if str(name) == 'status':
                list_quality_column.append(raw_data.iloc[:, i])
                list_quality_column_num.append(i)
            if str(name) == 'goal' and mode == 'classification':
                list_quality_column.append(raw_data.iloc[:, i])
                quality_column_goal = raw_data.iloc[:, i]
                list_quality_column_num.append(i)

        list_data_one_hot = []
        goal_one_hot = None

        for i, quality_column in enumerate(list_quality_column):
            list_data_one_hot.append(self.load_one_hot(quality_column, self.list_info_status[i]['possibility']))

        if quality_column_goal is not None:
            goal_one_hot = self.load_one_hot(quality_column_goal, self.info_goal['possibility'])

        result = raw_data.drop(raw_data.columns[list_quality_column_num], axis = 1)

        for i, data_one_hot in enumerate(list_data_one_hot):
            name = 'binaire_'+str(i)
            data_one_hot.columns = [name]*len(data_one_hot.columns)
            result = result.join(data_one_hot)

        if quality_column_goal is not None:
            name = 'goal'
            goal_one_hot.columns = [name]*len(goal_one_hot.columns)
            result = result.join(goal_one_hot)

        del list_data_one_hot

        return result


    def get_dataset(self):
        #nb_data_to_take = int(self.list_train_X.shape[0]*tol/100)
        return self.list_train_X[:]

    """def get_scaler_save(self):
        scaler_in_save = self.scaler_in.get_save()
        scaler_out_save = self.scaler_out.get_save()
        return scaler_in_save, scaler_out_save"""

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

    def create_input(self, raw_data, already_cleaned=False):
        # prend seulement les lignes qui vont être mise dans le modèle:
        data = raw_data.tail(self.config['lookback']).copy()

        data_cleaned = self.clean_data(data)

        data_formated = self.raw_data_to_data_formated(data_cleaned)

        # génére l'input à rentrer dans le model
        x = self.dataToDataset(data_formated, 0)

        last_x = x[0]
        len_predict = last_x.shape[1]
        last_x = np.reshape(last_x[0], (1, 1, len_predict))

        return last_x

    def load_one_hot(self, data_of_one_status, info_status):
        ds2 = pd.get_dummies(data_of_one_status)

        list_column_name_ds1 = info_status

        list_column_name_ds2 = list(ds2.columns.values)

        list_dif = list(set(list_column_name_ds1) - set(list_column_name_ds2))

        for dif in list_dif:
            ds2[dif] = 0

        ds2 = ds2.reindex(columns=list_column_name_ds1)

        return ds2

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

        # enlève la colonne 'timestamp'
        """if data_cleaned.columns.tolist().count('timestamp') != 0:
            del data_cleaned['timestamp']"""

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
        # génére les dataset
        """data_split = self.split_at_nan_value(data_cleaned)
        nb_data_split = len(data_split)
        print("Conversion des données en dataset...")
        bar = progressbar.ProgressBar(max_value=len(data_split)).start()
        for index, split in enumerate(data_split):
            if index != 0:
                train_X, train_y = self.dataToDataset(split, self.config['lookahead'])
                np.save("numpy/X/train_X_"+str(index), train_X)
                np.save("numpy/Y/train_y_"+str(index), train_y)
                del train_X
                del train_y
            else:
                train_X, train_y = self.dataToDataset(split, self.config['lookahead'], first_time= True)
                np.save("numpy/X/train_X_"+str(index), train_X)
                np.save("numpy/Y/train_y_"+str(index), train_y)
                del train_X
                del train_y
            bar.update(index)
        bar.finish()"""
        
        """for i in range(nb_data_split):
            if i == 0:
                self.list_train_X = np.load("numpy/X/train_X_"+str(i)+".npy")
                self.list_train_y = np.load("numpy/Y/train_y_"+str(i)+".npy")
            else:
                self.list_train_X = np.concatenate((self.list_train_X, np.load("numpy/X/train_X_"+str(i)+".npy")), axis=0)
                self.list_train_y = np.concatenate((self.list_train_y, np.load("numpy/Y/train_y_"+str(i)+".npy")), axis=0)

        # temporaire:
        # permet de faire des tests plus rapidement
        np.save("numpy/list_train_X", self.list_train_X)
        np.save("numpy/list_train_y", self.list_train_y)

        all_files = glob("numpy/X/*.npy")+glob("numpy/Y/*.npy")"""

