import progressbar
from pandas import read_csv, concat, DataFrame
from scaler import Scaler

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
        self.list_train_X = None
        self.list_train_y = None
        self.config = None
        self.nb_possibility = None
        self.info_goal = {}
        self.list_info_status = []

    def set(self, config, data):
        if data is not None:
            self.config = config
            self.create_dataset(data)
            self.dataset_is_ready = True
            self.config['list_info_status'] = self.list_info_status
            self.config['nb_entries'] = self.list_train_X.shape[2]
            self.config['nb_possibility'] = self.nb_possibility
            return self.config
        else:
            self.config = config
            self.scaler_in = Scaler(load = self.config['scaler_in_save'])
            self.scaler_out = Scaler(load = self.config['scaler_out_save'])

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


    def get_a_percent_of_dataset(self, percent_data):
        nb_data_to_take = int(self.list_train_X.shape[0]*percent_data/100)
        return self.list_train_X[:nb_data_to_take], self.list_train_y[:nb_data_to_take]

    def get_scaler_save(self):
        scaler_in_save = self.scaler_in.get_save()
        scaler_out_save = self.scaler_out.get_save()
        return scaler_in_save, scaler_out_save

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
            if str(column_format) == 'goal' and mode == 'classification':
                info = {'name':column_name, 'rank':rank}
                self.list_info_status.append(info)
                rank +=1

        # enlèvement des colonne 'disable' et trie par date si présence d'une colonne 'timestamp'
        data_cleaned = self.clean_data(data.copy())

        # enlève la colonne 'timestamp'
        if data_cleaned.columns.tolist().count('timestamp') != 0:
            del data_cleaned['timestamp']

        quality_column_goal = None
        list_quality_column = []
        list_quality_column_num = []
        for i, name in enumerate(data_cleaned.columns.tolist()):
            if str(name) == 'status':
                list_quality_column.append(data_cleaned.iloc[:, i])
                list_quality_column_num.append(i)
            if str(name) == 'goal' and mode == 'classification':
                list_quality_column.append(data_cleaned.iloc[:, i])
                quality_column_goal = data_cleaned.iloc[:, i]
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

        return data_cleaned

    def create_dataset(self, data):
        data_cleaned =self.set_data(data.copy())
        # génére les dataset
        data_split = self.split_at_nan_value(data_cleaned)
        nb_data_split = len(data_split)
        print("Conversion des données en dataset...")
        bar = progressbar.ProgressBar(max_value=len(data_split)).start()
        for index,split in enumerate(data_split):
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
        bar.finish()
        
        for i in range(nb_data_split):
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

        all_files = glob("numpy/X/*.npy")+glob("numpy/Y/*.npy")

    def split_at_nan_value(self, data):
        """convertie un dataFrame en plusieurs dataFrame qui ne sont jamais coupés par des cellules null
        """
        print("Split des données pour leur exploitation...")
        lookback = self.config['lookback']
        lookahead = self.config['lookahead']
        list_split = []
        list_temp = []
        first_element = 0
        current_element = 0
        bar_statut = 0
        bar = progressbar.ProgressBar(max_value=len(data.dropna())).start()
        # parcours les elements sans colonne null
        for index, row in data.dropna().iterrows():
            if first_element != 0:
                # si les numméro d'index se suivent
                if current_element == index:
                    current_element += 1
                    list_temp.append(index)
                else:
                    # si la liste d'index est assez grande pour être utile, elle est gardée:
                    if len(list_temp) > lookback+lookahead:
                        list_split.append(list_temp[:])
                    # recommence une liste
                    list_temp[:] = []
                    list_temp.append(index)
                    current_element = index+1
            # dans le premier cas de la boucle:
            else:
                first_element = index
                current_element = index+1
                list_temp.append(index)
            bar_statut+=1
            bar.update(bar_statut)
        bar.finish()
        if len(list_temp) > lookback+lookahead:
            list_split.append(list_temp[:])
        
        data_split = []
        for split in list_split:
            data_split.append(data.iloc[split,:])
        
        del list_temp
        del list_split
        return(data_split)

    def series_to_supervised(self, data, n_in=1, n_out=1, dropnan=True):
        """convert series to supervised learning
        """
        try:
            n_vars = 1 if type(data) is list else data.shape[1]
        except:
            n_vars = 1 if type(data) is list else data.shape[0]
        df = DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in-1, -1, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        # forecast sequence (t-1, t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
        # put it all together
        agg = concat(cols, axis=1)
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    def dataToDataset(self, train_pd, lookahead, first_time = False):
        # values_x = train_pd.loc[:, train_pd.columns != 'goal']
        values_x = train_pd
        values_y = train_pd['goal']

        lookback = self.config['lookback']
        mode = self.config['mode']
        nb_entries = len(values_x.columns)

        # frame as supervised learning
        reframed_x = self.series_to_supervised(values_x, lookback, 0)
        reframed_x.drop(reframed_x.tail(lookahead).index,inplace=True)

        values = reframed_x.values

        if lookahead != 0:
            reframed_y = self.series_to_supervised(values_y, 0, lookahead)
            reframed_y.drop(reframed_y.head(lookback-1).index,inplace=True)

            # mise en format ndarray
            goals = reframed_y.values

            # normalize features
            if first_time:
                self.scaler_in = Scaler(values)
                self.scaler_out = Scaler(goals, line=True)
            
            scaled = self.scaler_in.rescale(values)
            goals = self.scaler_out.rescale(goals)

            train_X, train_y = scaled, goals
            train_X = train_X.reshape(train_X.shape[0], 1, nb_entries*lookback)

            self.nb_possibility = int(train_y.shape[1]/lookahead)

            if mode == 'classification':
                train_y = train_y.reshape(train_y.shape[0], lookahead, self.nb_possibility)
            else:
                train_y = train_y.reshape(train_y.shape[0], 1, lookahead)

            del values
            del reframed_x
            del reframed_y
            del goals
            return train_X, train_y
        else:
            del reframed_x
            scaled = self.scaler_in.rescale(values)
            return scaled.reshape(scaled.shape[0], 1, nb_entries*lookback)
{"mode":"full","isActive":false}