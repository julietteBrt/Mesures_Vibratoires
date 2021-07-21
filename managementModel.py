from kmeans import KMeansClustering
from dataset import Dataset
from datetime import timedelta, date, datetime
from functools import reduce
import plotly.graph_objects as go
import pandas as pd
import pickle
import numpy as np
import os
import json

class ManagementModel:
    def __init__(self):
        self.dataset = Dataset()
        self.clustering_model = KMeansClustering()
        self.config = None
        self.name_goal = None

    def set_config(self, config):#, data):
        self.config = config
       

    def set_dataset(self, data):
        # formatte les données pour les envoyer au modèle
        # récupère la configuration du format pour la sauvegarder
        self.config = self.dataset.set(self.config, data)

    def set_model(self):
        self.clustering_model.set(self.config)

    def train(self):
        data = self.dataset.data.copy()
        data = self.dataset.set_data(data)
        list_format_column = self.config['list_format_column']
        index_vibration = list_format_column.index("vibration")

        if 'timestamp' in list_format_column:
            index_goal = list_format_column.index("timestamp")
            formats = ['%Y-%m-%d %H:%M:%S.%f', '%d/%m/%Y %H:%M','%m/%d/%Y  %H:%M:%S', '%d-%m-%y', '%d-%m-%Y', '%Y-%m-%d', '%b %y', '%B %Y', '%d %B %Y']
            data['timestamp'] = reduce(lambda l, r: l.combine_first(r), [pd.to_datetime(data['timestamp'], format=fmt, errors='coerce') for fmt in formats])
        
        self.clustering_model.train(self.config, data[['vibration']])

    def demo(self, data_in):

        data = data_in.copy()

        data = data.reset_index(drop=True)
        
        list_format_column = self.config['list_format_column']
        index_vibration = list_format_column.index("vibration")

        if 'timestamp' in list_format_column:
            index_goal = list_format_column.index("timestamp")
            formats = ['%Y-%m-%d %H:%M:%S.%f', '%d/%m/%Y %H:%M','%m/%d/%Y  %H:%M:%S', '%d-%m-%y', '%d-%m-%Y', '%Y-%m-%d', '%b %y', '%B %Y', '%d %B %Y']
            data.iloc[:, index_goal] = reduce(lambda l, r: l.combine_first(r), [pd.to_datetime(data.iloc[:, index_goal], format=fmt, errors='coerce') for fmt in formats])

            last_date_str = data.iloc[[-1],index_goal].values[0]
            last_date = last_date_str

            liste_date_all = []
            for index, row in data.iterrows():
                dat_str = row[index_goal]
                date  = dat_str
                liste_date_all.append(date)

        else:
            liste_date_all = list(range(len(data)))

        data = self.dataset.set_data(data)
        fig = go.Figure()
        df_results = self.clustering_model.fit_predict(data[['vibration']])
        dico_cst = self.min_max_mean(df_results)
        fig.add_trace(go.Scatter(x=liste_date_all, y=data['vibration'], mode = 'markers', marker = dict(color = df_results['cluster'], colorscale = 'Viridis')))
        fig.add_hline(y=dico_cst['max'], line_color='red')
        fig.add_hline(y=dico_cst['min'], line_color='red')
        fig.add_hline(y=dico_cst['low_mean'], line_color='green')
        fig.add_hline(y=dico_cst['high_mean'], line_color='green')
        fig.show()
        return data

    def min_max_mean(self, data):
        dico = {}
        clusters = data['cluster'].value_counts().copy()
        clusters = clusters.sort_values(ascending=False)
        clusters = clusters.reset_index()
        clusters['cluster'].sum()
        index = 0
        nb_points = 0
        for i in range(len(clusters)):
            if(nb_points < clusters['cluster'].sum()*0.75):
                nb_points += clusters.loc[i, 'cluster']
        else:
            index = clusters.loc[i, 'index']
        cluster1 = clusters.loc[index-1, :]
        cluster2 = clusters.loc[index, :]

        mean_cluster1 = data[data['cluster'] == cluster1['index']]['vibration'].mean()
        mean_cluster2 = data[data['cluster'] == cluster2['index']]['vibration'].mean()

        if (mean_cluster1 > mean_cluster2):
            max_vib = data[data['cluster'] == cluster1['index']]['vibration'].max()
            min_vib = data[data['cluster'] == cluster2['index']]['vibration'].min()
            dico['low_mean'] = mean_cluster2
            dico['high_mean'] = mean_cluster1
            dico['max'] = max_vib
            dico['min'] = min_vib
        else:
            max_vib = data[data['cluster'] == cluster2['index']]['vibration'].max()
            min_vib = data[data['cluster'] == cluster1['index']]['vibration'].min()
            dico['low_mean'] = mean_cluster1
            dico['high_mean'] = mean_cluster2
            dico['max'] = max_vib
            dico['min'] = min_vib
        return dico

    def calcul(self, data):
        x = self.dataset.create_input(data)
        return True

    def save(self):
        # Sauvegarde des configurations du modèle
        if self.clustering_model.model_is_trained:
            name_file = self.config['n_clusters']
            with open(f'model/{name_file}.pkl', 'wb') as file:
                pickle.dump(self.clustering_model, file)
        else:
            print("Aucun modèle à sauvegarder.")

    def load(self, config, data, model):
        del self.dataset
        del self.clustering_model

        self.dataset = Dataset()
        self.clustering_model = KMeansClustering()

        print("Chargement du model")

        self.set_config(config, data)
        self.set_dataset(data)

        self.clustering_model.load(config, model)

        return True
