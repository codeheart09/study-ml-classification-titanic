# Core
import os
from time import time
from datetime import datetime

# Libraries
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 12)
pd.set_option('display.width', 350)
pd.set_option('display.max_colwidth', 12)


def time_ms():
    return time() * 1000


def date():
    return datetime.now()


def get_dataset():
    csv_path_train = os.path.join('data', 'train.csv')
    csv_path_test = os.path.join('data', 'train.csv')

    train_set = pd.read_csv(csv_path_train)
    test_set = pd.read_csv(csv_path_test)

    return train_set, test_set


def display_data_info(dataset):
    print('Dataset head:')
    print(dataset.head(), '\n')

    print('Dataset info:')
    print(dataset.info(), '\n')

    print('Dataset describe:')
    print(dataset.describe(), '\n')


def plot_histograms(dataset):
    print('Plotting dataset histograms...', end=' ')
    dataset.hist(bins=20, figsize=(20, 15), column=['Pclass', 'Age', 'SibSp', 'Parch', 'Fare'])
    plt.show()
    print('done!', '\n')


def analyse_data_correlations(dataset):
    print('Data correlations:')
    correlation_matrix = dataset.corr()
    print(correlation_matrix['Survived'].sort_values(ascending=False), '\n')

    print('Plotting correlations...', end=' ')
    scatter_matrix(dataset[['Survived', 'Fare', 'Parch', 'Age', 'Pclass']], figsize=(12, 8))
    plt.show()
    print('done!', '\n')


def separe_predictors_labels(dataset):
    print('Separating predictors from labels...', end=' ')
    predictors = dataset.drop('Survived', axis=1)
    labels = dataset['Survived'].copy()
    print('done!', '\n')
    return predictors, labels
