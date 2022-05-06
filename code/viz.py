import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


sns.set_theme(style="darkgrid")
from preprocess import csvToDataFrame, label_to_plot
from sklearn.decomposition import PCA

sns.set_theme(style="darkgrid")
sns.set(rc={'figure.figsize':(11.7,8.27)})


def plot_expected_vs_actual(predicted, actual):
    '''
    Plot for the expected vs actual price. 
    Use this when you want to see the predictions vs the actual price
    for a single model

    :param predicted: the predictions the model made
    :param actual: the actual price of the trip
    '''
    # plot the expected vs actual
    plt.figure(figsize=(10,5))
    ax = plt.axes()
    ax.set_title('Expected vs Actual')
    ax.set_xlabel('Day in February')
    ax.set_ylabel('Trip Total Price in dollars')
    x_pre, y_pre = label_to_plot(predicted)
    x_act, y_act = label_to_plot(actual)
    ax.set_xticks(x_pre)
    ax.set_yticks(np.arange(13, 35, 1))
    plt.plot(y_pre, label='Predicted Price')
    plt.plot(y_act, label='Actual Price')
    plt.legend()
    plt.show()

def plot_data_distribution(data_to_plot, num_components = 3):
    '''
    Function to plot the distribution of data points

    :param data_to_plot: the data to plot
    :param num_components: the number of components to use for PCA
    '''

    if num_components != 3 and num_components != 2:
        raise ValueError("num_components must be either 2 or 3")

    # Set up for PCA object
    pca = PCA(n_components = num_components)
    data_pca = pca.fit_transform(data_to_plot)

    # Set up for 3d plotting
    if num_components == 3:
        plt.style.use('seaborn-whitegrid')
        fig = plt.figure(figsize = (10,6))
        ax = fig.gca(projection='3d')
        c_map = plt.cm.get_cmap('jet', 7) # Change 21 according to the number of labels we have
        ax.scatter(data_pca[:, 0], data_pca[:, 1], data_pca[:, 2],
                    cmap = c_map , c = data_to_plot["trip_total"])
        
        summary = "3d plot of the data points in the joint dataset \n \
            The color of the points represents the class of the data point"
        norm = mpl.colors.Normalize(vmin=0, vmax=6)
        sm = plt.cm.ScalarMappable(cmap=c_map, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, label = "Class")
        plt.xlabel('PC-1', fontsize = 12) , plt.ylabel('PC-2', fontsize=12),ax.set_zlabel('PC-3', fontsize=12), plt.title('PCA of our Dataset where PCA = 3. \n', fontsize = 20)
        fig.text(.5, .025, summary, ha='center', fontsize = 'xx-large')
        plt.show()
    else:
        plt.style.use('seaborn-whitegrid')
        fig = plt.figure(figsize = (10,6))
        c_map = plt.cm.get_cmap('jet', 7) # Change 21 according to the number of labels we have
        ax.scatter(data_pca[:, 0], data_pca[:, 1],
                    cmap = c_map , c = data_to_plot["trip_total"])
        
        summary = "2d plot of the data points in the joint dataset \n \
            The color of the points represents the class of the data point"
        norm = mpl.colors.Normalize(vmin=0, vmax=6)
        sm = plt.cm.ScalarMappable(cmap=c_map, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, label = "Class")
        plt.xlabel('PC-1', fontsize = 12) , plt.ylabel('PC-2', fontsize=12), plt.title('PCA of our Dataset where PCA = 2. \n', fontsize = 20)
        fig.text(.5, .025, summary, ha='center', fontsize = 'xx-large')
        plt.show()

def plot_result(pred_reg, pred_kmeans, pred_lstm, true):
    '''
    Function to plot the results of our three models

    :param pred_reg: the predicted result of the regression model
    :param pred_kmeans: the predicted result of the kmeans model
    :param pred_lstm: the predicted result of the lstm model
    :param true: the true result of the dataset
    '''

    plt.figure(figsize=(10,5))
    ax = plt.axes()
    ax.set_title('Model Results vs Actual')
    ax.set_xlabel('Day in February')
    ax.set_ylabel('Trip Total Price in dollars')
    x_reg, y_reg = label_to_plot(pred_reg)
    x_kmeans, y_kmeans = label_to_plot(pred_kmeans)
    x_lstm, y_lstm = label_to_plot(pred_lstm)
    x_act, y_act = label_to_plot(true)
    ax.set_xticks(x_reg)
    ax.set_yticks(np.arange(13, 35, 1))
    plt.plot(y_reg, label='Logistic Regression')
    plt.plot(y_kmeans, label='Kmeans+SVM')
    plt.plot(y_lstm, label='LSTM')
    plt.plot(y_act, label='Actual Price')
    plt.legend()
    plt.show()