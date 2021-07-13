import os
from datetime import datetime
import sys
import random
import json
from pandas import DataFrame
from PyQt5.QtWidgets import (
    QApplication, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QFileDialog,
    QTableWidget, QHeaderView, QTableWidgetItem, QLabel, QTextEdit, QCheckBox
)
from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSlot
# from PyQt5.QtGui import QIconconda

import plotly.graph_objects as go

import pandas as pd
import numpy as np

#from managementModel import ManagementModel

def csv_to_pandas(fileName):
    with open(fileName) as f:
        firstLine = f.readline()
    nbComma = firstLine.count(',')
    nbSemicolon = firstLine.count(';')
    if nbComma>nbSemicolon:
        dataset = pd.read_csv(fileName)
    else:
        dataset = pd.read_csv(fileName, sep =";")
    return dataset

class App(QWidget):
    def __init__(self):
        self.nb_train = 0
        
        super().__init__()
        self.left = 100
        self.top = 100
        self.title = 'Réglage du modèle de Clustering'
        self.width = 1500
        self.height = 1000
        self.init_UI()
        self.dialogWindow3 = None
        #self.management_model = ManagementModel()
        
    def init_UI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        
        self.dialog = QFileDialog()
        self.layout = QVBoxLayout()
        
        self.reglage_box = QHBoxLayout()
        self.resultat_box = QVBoxLayout()
        
        self.reglage_box.addStretch()
        self.resultat_box.addLayout(self.reglage_box)
        
        self.setLayout(self.resultat_box)
        
        self.resultat_box.addWidget(self.dialog)
        self.setLayout(self.layout)
        
        # Connecting the signal
        self.dialog.fileSelected.connect(self.drawSetData)
        self.dialog.fileSelected.connect(self.drawSetTrain)
        
        # permettra d'avoir plusieurs fenetres
        self.dialogs = list()
        
        self.show()
        
    @pyqtSlot()
    def selectSaveModel(self):
        self.dialogs[:] = []
        dialogWindow = Second(self)
        self.dialogs.append(dialogWindow)
        dialogWindow.show()
        
    @pyqtSlot()
    def loadSaveModel(self):
        """ Récuperation du .json de configuration
        """
        self.textbox_nb_epoch.setText("0")
        #permet de charger une sauvegarde pour le lstm, pas le embedding
        self.dialogs[0].hide()
        self.name_file_config = str(self.dialogs[0].dialog2.selectedFiles())[2:-2]

        with open(self.name_file_config) as json_file: 
            config = json.load(json_file)

        name_file = config['name_goal']+'.h5'
        dir = os.path.dirname(self.name_file_config)
        name_file_model = dir+'/'+name_file

        model = 'hello' #tf.keras.models.load_model(name_file_model)

        lookback = config['lookback']
        lookahead = config['lookahead']
        percent_data = config['percent_data']
        nb_layers = config['nb_layers']
        list_format_column = config['list_format_column']
        mode = config['mode']

        self.old_selection = {'list_format_column':list_format_column[:], 'lookback':lookback, 'lookahead':lookahead, 'nb_layers':nb_layers, 'mode': mode}
        self.old_percent_data = None

        self.textbox_lookback.setText(str(lookback))
        self.textbox_lookahead.setText(str(lookahead))
        self.textbox_percent_data.setText(str(percent_data))
        self.textbox_nb_layers.setText(str(nb_layers))

        if mode == 'classification':
            self.checkbox_classification.setChecked(True)
        else:
            self.checkbox_classification.setChecked(False)

        for i in range(len(self.listComboBox)):
            choose = list_format_column[i]
            index_combo_box = self.dict_items_selectable[choose]
            self.listComboBox[i].setCurrentIndex(index_combo_box)

        del self.management_model
        #self.management_model = ManagementModel()
        self.management_model.set_config(config, self.data)
        self.management_model.set_dataset(self.data)
        self.management_model.set_model()

        self.nb_train = 1

    @pyqtSlot()
    def setTrain(self):
        """recuperation des données de la l'interface
        """
        list_format_column = []
        for comboBox in self.listComboBox:
            list_format_column.append(comboBox.currentText())

        if list_format_column.count('goal') == 0:
            print("Aucune colonne goal n'a été sélectionnée")
        else:
            lookback = int(self.textbox_lookback.toPlainText())
            lookahead = int(self.textbox_lookahead.toPlainText())+1
            nb_epoch = int(self.textbox_nb_epoch.toPlainText())
            percent_data = int(self.textbox_percent_data.toPlainText())
            nb_layers = int(self.textbox_nb_layers.toPlainText())
            classification = self.checkbox_classification.isChecked()

            if classification == True:
                mode = 'classification'
            else:
                mode = 'lstm'
            
            selection = {'list_format_column':list_format_column[:], 'lookback':lookback, 'lookahead':lookahead, 'nb_layers':nb_layers, 'mode': mode}

            config = {
                'list_format_column':list_format_column[:],
                'lookback':lookback,
                'lookahead':lookahead,
                'nb_layers':nb_layers,
                'percent_data': percent_data,
                'mode': mode,
                'nb_entries': None
            }
            
            #si c'est la première fois que l'on structure le ML
            if self.nb_train == 0:
                self.management_model.set_config(config, self.data)
                self.management_model.set_dataset(self.data)
                self.management_model.set_model()
                self.management_model.train(nb_epoch)
                self.management_model.save()
            elif self.old_selection == selection and percent_data == self.old_percent_data:
                # dans le cas où aucun réglages ne modifie la structure du ML
                # ne lance que l'apprentissage
                self.management_model.train(nb_epoch)
            elif self.old_selection == selection:
                # dans le cas ou rien ne change sauf le pourcentage ou le nombre d'epoch
                if self.old_percent_data != percent_data:
                    self.management_model.set_percent(percent_data)
                if nb_epoch > 0:
                    self.management_model.train(nb_epoch)
                    self.management_model.save()
            else:
                self.nb_train = 0
                self.management_model.set_config(config, self.data)
                self.management_model.set_dataset(self.data)
                self.management_model.set_model()
                self.management_model.train(nb_epoch)
                self.management_model.save()
            
            # enregistre le format pour le comparer avec le prochain apprentissage
            self.old_selection = {'list_format_column':list_format_column[:], 'lookback':lookback, 'lookahead':lookahead, 'nb_layers':nb_layers, 'mode': mode}
            self.old_percent_data = percent_data
            
            if self.management_model.lstm_model.model_is_set:
                data_out_pd = self.management_model.demo(self.data,percent_data)
                data_out_pd.to_csv('temp/out.csv', sep = ';')

            self.nb_train += 1

    @pyqtSlot()
    def saveModel(self):
        """sauvegarde d'un model pour l'utiliser plus tard
        """
        formats = "Hierarchical Data Format (HDF) (*.h5)"
        output_file = QFileDialog.getSaveFileName(self, "Save as model", "untitled.h5", formats)
        output_file = str(output_file).strip()
        
        if len(output_file) == 0:
            return

        self.management_model.save()

    @pyqtSlot()
    def cleanDataAndSet(self):
        # anlève l'affichage du dataset précédant
        self.layout.removeWidget(self.table)
        self.table.deleteLater()
        self.table = None

        self.layout.removeWidget(self.format)
        self.format.deleteLater()
        self.format = None

        # reset qwidget de selection de fichier

        self.layout.removeWidget(self.dialog)
        self.dialog.deleteLater()
        self.dialog = None

        # met en selection le nouveau qwidget

        self.dialog = self.dialogWindow3.dialog3

        # dessine l'UI avec le nouveau dataset

        self.drawSetData()

        # ferme la selection de fichier

        self.dialogWindow3.close()

        # oblige de recréer un model de machine learning

        self.nb_train = 0

    @pyqtSlot()
    def drawSetData(self):
        """dessine le tableau csv de données afin de choisir les entrées et sorties
        """

        self.data = csv_to_pandas(str(self.dialog.selectedFiles())[2:-2])
        list_name_column = self.data.columns.values.tolist()
        
        echantillon = self.data.head()
        
        self.table = QTableWidget()
        self.table.setColumnCount(len(list_name_column))
        self.table.setHorizontalHeaderLabels(list_name_column)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        self.resultat_box.addWidget(self.table)
        
        listColumn = []
        self.items = 0
        
        for nameColumn in list_name_column:
            listColumn.append(echantillon[nameColumn].tolist())
        
        # affichage des 5 premières valeurs du csv 
        listElement = []
        for index in range(5):
            listElement.append([])
            for column in listColumn:
                listElement[index].append(QTableWidgetItem(str(column[index])))
            self.table.insertRow(self.items)
            for index2, column in enumerate(listColumn, start=0):
                self.table.setItem(self.items, index2, listElement[index][index2])
            self.items += 1
        
        # choix format de lecture des colonnes
        self.format = QTableWidget()
        self.format.setColumnCount(len(list_name_column))
        self.format.setHorizontalHeaderLabels(list_name_column)
        self.format.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.format.setFixedHeight(120)
        
        
        # permet de selectionner les types des colonnes
        listFormat = []
        self.listComboBox = []
        for column in listColumn:
            listFormat.append(QtWidgets.QTableWidgetItem())
        self.format.insertRow(0)
        for index, column in enumerate(listColumn, start=0):
            self.format.setItem(0, index, listFormat[index])
            comboBox = QtWidgets.QComboBox()
            comboBox.addItems(["disable", "number", "status", "timestamp"])
            self.listComboBox.append(comboBox)
            self.format.setCellWidget(0, index, self.listComboBox[index])
        self.resultat_box.addWidget(self.format)
        self.resultat_box.addWidget(self.table)

        self.dict_items_selectable = {"disable":0, "number":1, "status":2, "timestamp":3}

    @pyqtSlot()
    def load_an_other_csv(self):
        # ouverture de la fenetre de selection de csv
        if self.dialogWindow3 is not None:
            self.dialogWindow3 = None

        self.dialogWindow3 = Third(self)
        self.dialogs.append(self.dialogWindow3)
        self.dialogWindow3.show()

    @pyqtSlot()
    def drawSetTrain(self):
        """affichage pour les réglages
        """
        self.label_lookback = QLabel("# Clusters")
        self.textbox_lookback = QTextEdit()
        self.textbox_lookback.setFixedHeight(25)
        self.label_lookahead = QLabel("# Runs")
        self.textbox_lookahead = QTextEdit()
        self.textbox_lookahead.setFixedHeight(25)
        self.label_nb_epoch = QLabel("Max # iterations")
        self.textbox_nb_epoch = QTextEdit()
        self.textbox_nb_epoch.setFixedHeight(25)
        self.label_percent_data = QLabel("Tolerance")
        self.textbox_percent_data = QTextEdit()
        self.textbox_percent_data.setFixedHeight(25)
        """self.label_nb_layers = QLabel("Number of layers")
        self.textbox_nb_layers = QTextEdit()
        self.textbox_nb_layers.setFixedHeight(25)
        self.checkbox_classification = QCheckBox("Classification")
        self.checkbox_classification.setFixedHeight(25)"""
        
        self.reglage_box.addWidget(self.label_lookback)
        self.reglage_box.addWidget(self.textbox_lookback)
        self.reglage_box.addWidget(self.label_lookahead)
        self.reglage_box.addWidget(self.textbox_lookahead)
        self.reglage_box.addWidget(self.label_nb_epoch)
        self.reglage_box.addWidget(self.textbox_nb_epoch)
        self.reglage_box.addWidget(self.label_percent_data)
        self.reglage_box.addWidget(self.textbox_percent_data)
        #self.reglage_box.addWidget(self.label_nb_layers)
        """self.reglage_box.addWidget(self.textbox_nb_layers)
        self.reglage_box.addWidget(self.checkbox_classification)"""
        
        self.textbox_lookback.setText("7")
        self.textbox_lookahead.setText("30")
        self.textbox_nb_epoch.setText("3")
        self.textbox_percent_data.setText("95")
        #self.textbox_nb_layers.setText("3")
        
        self.button = QPushButton("Run training")
        #self.button_load = QPushButton("Load .json")
        self.button_csv = QPushButton("Load an other csv")
        self.reglage_box.addWidget(self.button)
        #self.reglage_box.addWidget(self.button_load)
        self.reglage_box.addWidget(self.button_csv)
        
        self.button.clicked.connect(self.setTrain)
        #self.button_load.clicked.connect(self.selectSaveModel)
        self.button_csv.clicked.connect(self.load_an_other_csv)


class Second(QWidget):
    """Fenetre qui s'ouvre pour charger le .pkl
    """
    def __init__(self, parent=None):
        super().__init__()
        self.left = 100
        self.top = 100
        self.title = 'Select .pkl file'
        self.width = 1500
        self.height = 1000
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        
        self.dialog2 = QFileDialog()
        self.layout = QVBoxLayout()
        
        self.reglage_box = QHBoxLayout()
        self.resultat_box = QVBoxLayout()
        
        self.reglage_box.addStretch()
        self.resultat_box.addLayout(self.reglage_box)
        
        self.setLayout(self.resultat_box)
        
        self.resultat_box.addWidget(self.dialog2)
        self.setLayout(self.layout)
        
        self.dialog2.fileSelected.connect(parent.loadSaveModel)
        
        self.show()

class Third(QWidget):
    """Fenetre qui s'ouvre pour changer le .csv
    """
    def __init__(self, parent=None):
        super().__init__()
        self.left = 100
        self.top = 100
        self.title = 'Select .csv file'
        self.width = 1500
        self.height = 1000
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        
        self.dialog3 = QFileDialog()
        self.layout = QVBoxLayout()
        
        self.reglage_box = QHBoxLayout()
        self.resultat_box = QVBoxLayout()
        
        self.reglage_box.addStretch()
        self.resultat_box.addLayout(self.reglage_box)
        
        self.setLayout(self.resultat_box)
        
        self.resultat_box.addWidget(self.dialog3)
        self.setLayout(self.layout)
        
        self.dialog3.fileSelected.connect(parent.cleanDataAndSet)
        
        self.show()


if __name__ == '__main__':
    app = QApplication([])
    ex = App()
    sys.exit(app.exec_())
{"mode":"full", "isActive":False}