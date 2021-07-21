import sklearn
from sklearn.cluster import KMeans
import pickle 
import datetime

import os

class KMeansClustering:
    def __init__(self):
        self.model_is_set = False
        self.model_is_trained = False

        self.model = None
        self.config = None
        self.name_file_pkl_save = None

    def set(self, config):
        self.config = config
        n_clusters = self.config['n_clusters']
        n_init = self.config['n_init']
        max_iter = self.config['max_iter']
        tol = self.config['tol']

        self.model = KMeans(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter, tol=tol, random_state=42)
        self.model_is_set = True

    def load(self, config, model):
        self.config = config
        print("Chargement du model")
        self.model = model
        self.model_is_set = True
        self.model_is_trained = True
        self.model.summary()

    def train(self, config, train_x):
        self.config = config
        ncluster_silhouette = []

        for i in range(2, 10):
            kmeans = KMeans(n_clusters=i, random_state=42)
            kmeans.fit_predict(train_x)
            score = sklearn.metrics.silhouette_score(train_x, kmeans.labels_, metric='euclidean')
            ncluster_silhouette.append(score)

        # entraîner kmeans avec le nombre de clusters optimal
        self.config = config
        n_clusters = self.get_best_k_with_silhouette(ncluster_silhouette)
        n_init = self.config['n_init']
        max_iter = self.config['max_iter']
        tol = self.config['tol']

        self.model = KMeans(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter, tol=tol, random_state=42)
        
        # sauvegarder le modèle
        self.name_file_pkl_save = f'model/{datetime.datetime.now().date()}_model_with_{n_clusters}_clusters.pkl'
        with open(self.name_file_pkl_save, 'wb') as file:
            pickle.dump(self.model, file)
        
        self.model_is_trained = True
        return self.model

    def get_best_k_with_silhouette(self, ncluster_silhouette):
        for i in range(len(ncluster_silhouette)):
            if (i > 0) and (ncluster_silhouette[i] > ncluster_silhouette[i-1]) and (ncluster_silhouette[i] > ncluster_silhouette[i+1]):
                k_opti = i + 2
                self.config['n_clusters'] = k_opti
                return k_opti
                break
        return 0

    def fit_predict(self, x):
        print(f'x: {x}')
        x['cluster'] = self.model.fit_predict(x[['vibration']])
        return x

    def predict(self, x):
        print(f'x: {x}')
        x['cluster'] = self.model.predict(x[['vibration']])
        return x