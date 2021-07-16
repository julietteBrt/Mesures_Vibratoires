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

    """def get_name_goal(self, data):
        list_format_column = self.config['list_format_column']
        index_goal = list_format_column.index("goal")
        self.name_goal = data.columns.tolist()[index_goal]
        self.config['name_goal'] = self.name_goal"""

    def set_config(self, config):#, data):
        self.config = config
        """if data is not None:
            self.get_name_goal(data.head(1))"""

    """def set_percent(self, percent):
        self.config['percent'] = percent
        self.dataset.config['percent'] = percent
        self.clustering_model.config['percent'] = percent"""

    def set_dataset(self, data):
        # formatte les données pour les envoyer au modèle
        # récupère la configuration du format pour la sauvegarder
        self.config = self.dataset.set(self.config, data)

    def set_model(self):
        self.clustering_model.set(self.config)

    def train(self):
        data = self.dataset.data.copy()
        data = self.dataset.set_data(data)
        #print(f'MANAMODEL: {type(data)}')
        list_format_column = self.config['list_format_column']
        index_vibration = list_format_column.index("vibration")

        if 'timestamp' in list_format_column:
            index_goal = list_format_column.index("timestamp")
            formats = ['%Y-%m-%d %H:%M:%S.%f', '%m/%d/%Y  %H:%M:%S', '%d-%m-%y', '%d-%m-%Y', '%Y-%m-%d', '%b %y', '%B %Y', '%d %B %Y']
            #print(data)
            data['timestamp'] = reduce(lambda l, r: l.combine_first(r), [pd.to_datetime(data['timestamp'], format=fmt, errors='coerce') for fmt in formats])
            #print(f'type: {type(data)}')
        #print(data.dtypes)
        self.clustering_model.train(self.config, data[['vibration']])

    def demo(self, data_in):

        data = data_in.copy()

        data = data.reset_index(drop=True)
        
        """x = self.dataset.create_input(data, already_cleaned=True)

        out = self.clustering_model.calcul(x)
        out_unscalded = self.dataset.scaler_out.unscale(out)

        y = np.reshape(out_unscalded, out_unscalded.shape[2])"""

        list_format_column = self.config['list_format_column']
        index_vibration = list_format_column.index("vibration")

        if 'timestamp' in list_format_column:
            index_goal = list_format_column.index("timestamp")
            formats = ['%Y-%m-%d %H:%M:%S.%f', '%m/%d/%Y  %H:%M:%S', '%d-%m-%y', '%d-%m-%Y', '%Y-%m-%d', '%b %y', '%B %Y', '%d %B %Y']
            data.iloc[:, index_goal] = reduce(lambda l, r: l.combine_first(r), [pd.to_datetime(data.iloc[:, index_goal], format=fmt, errors='coerce') for fmt in formats])

            """data.iloc[:, index_goal] =  pd.to_datetime(data.iloc[:, index_goal], errors='coerce', dayfirst=True)
            data.iloc[:, index_goal] =  pd.to_datetime(data.iloc[:, index_goal], errors='coerce', dayfirst=True)

            data.iloc[:, index_goal] = data.iloc[:, index_goal].dt.strftime('%m/%d/%Y  %H:%M:%S')
            data.iloc[:, index_goal] = data.iloc[:, index_goal].dt.strftime('%m/%d/%Y  %H:%M:%S')"""

            last_date_str = data.iloc[[-1],index_goal].values[0]
            last_date = last_date_str

            liste_date_all = []
            for index, row in data.iterrows():
                dat_str = row[index_goal]
                date  = dat_str
                liste_date_all.append(date)

            """#calcul de l'ecart entre deux dates
            delta_date_secondes = abs((liste_date_all[1] - liste_date_all[0]).seconds)
            delta_date_days = abs((liste_date_all[1] - liste_date_all[0]).days)

            liste_date_prediction = []
            for i in range(self.config['lookahead']):
                if delta_date_days == 0:
                    liste_date_prediction.append(last_date + timedelta(seconds=i*delta_date_secondes))
                else:
                    liste_date_prediction.append(last_date + timedelta(days=i*delta_date_days))

            liste_date_entree = []
            for index, row in data.iterrows():
                dat_str = row[index_goal]
                date  = datetime.strptime(dat_str, '%m/%d/%Y  %H:%M:%S')
                liste_date_entree.append(date)
"""
        else:
            #last_date_str = data.shape[0]

            #debut_x = len(data)
            #fin_x = size_wanted

            #liste_date_prediction = list(range(debut_y,fin_y))
            #liste_date_entree = list(range(debut_x,fin_x))

            liste_date_all = list(range(len(data)))

        data = self.dataset.set_data(data)
        fig = go.Figure()
        #Ici y : vibrations, les points seront à colorer selon leur cluster
        df_results = self.clustering_model.predict(data[['vibration']])
        print(df_results)
        fig.add_trace(go.Scatter(x=liste_date_all, y=data['vibration'], name = 'courbe réelle', mode = 'markers', marker = dict(color = df_results['cluster'], colorscale = 'Viridis')))
        #fig.add_trace(go.Scatter(x=liste_date_prediction, y=y ,name = 'Prédiction', connectgaps=True))
        #fig.add_trace(go.Scatter(x=liste_date_entree, y=data[self.name_goal], name = 'données utilisées pour faire la prédiction', connectgaps=True))
        fig.show()

        """df_y_values = pd.DataFrame(y)
        df_date = pd.DataFrame(liste_date_prediction)

        df_y = pd.concat([df_date, df_y_values], axis=1).reindex(df_date.index)

        del df_y_values
        del df_date

        df_y.columns = ['timestamp', 'goal']

        # we rename columns,
        # because we can have trouble if some colmuns have the same name"""
        """i = 0
        list_temp_colomn_name = []
        for col in data.columns: 
            if col == 'number':
                list_temp_colomn_name.append('number'+str(i))
                i+=1
            elif col == 'status':
                list_temp_colomn_name.append('status'+str(i))
                i+=1
            else:
                list_temp_colomn_name.append(col)

        data.columns =list_temp_colomn_name

        if 'timestamp' in data.columns.tolist():
            #data['timestamp'] =  pd.to_datetime(data['timestamp'], format='%Y-%m-%d')
            #data['timestamp'] =  pd.to_datetime(data['timestamp'], errors='coerce')

            #data['timestamp'] = data['timestamp'].dt.strftime('%m/%d/%Y  %H:%M:%S')

            result = data.append(df_y, sort=False)
            result.sort_values(by='timestamp')
        else:
            result = data.append(df_y, sort=False)

        result = result.reset_index(drop=True)

        return result"""
        return data

    def calcul(self,data):
        x = self.dataset.create_input(data)

        """out = self.clustering_model.calcul(x)
        out_unscalded = self.dataset.scaler_out.unscale(out)

        result = out_unscalded.flatten()
        return result"""
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
