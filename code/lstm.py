import numpy as np
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from preprocess import preprocess
from viz import plot_data_distribution, plot_expected_vs_actual
from sklearn.model_selection import TimeSeriesSplit


class Lstm(object):

    def __init__(self, filename, filename_unseen):
        '''
        # Build our model here using keras Sequential()
        # Activation function: sigmoid
        # Loss function: mean squared error
        # Optimizer: adam
        # 1 visible layer with a hidden layer consisting of 4 LSTM blocks


        :param filename: the filename of the training data
        :param filename_unseen: the filename of the unseen data
        
        '''
        _,_,_,_, self.unseen_data, self.unseen_label, self.df, self.df_label = preprocess(filename, filename_unseen)
        self.look_back_val = 1

        scaler = MinMaxScaler(feature_range=(0, 1))
        self.df_label = scaler.fit_transform(self.df_label.reshape(-1, 1))
        self.unseen_label = scaler.transform(self.unseen_label.reshape(-1, 1))
        # change dataset to reflect the look_back_val
        self.trainX, self.trainY = self.create_dataset(self.df_label, self.look_back_val)
        self.testX, self.testY = self.create_dataset(self.unseen_label, self.look_back_val)

        # reshape input to be [samples, time steps, features]
        self.trainX = np.reshape(self.trainX, (self.trainX.shape[0], 1, self.trainX.shape[1]))
        self.testX = np.reshape(self.testX, (self.testX.shape[0], 1, self.testX.shape[1]))
        

        model = Sequential()
        model.add(LSTM(4, input_shape=(1, self.look_back_val), return_sequences=False))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(self.trainX, self.trainY, epochs=100, batch_size=1, verbose=2)
    
        # make predictions
        self.trainPredict = model.predict(self.trainX)
        self.testPredict = model.predict(self.testX)

        # invert predictions
        self.trainPredict = scaler.inverse_transform(self.trainPredict).reshape(-1, 1)
        trainY = scaler.inverse_transform(self.trainY).reshape(-1, 1)
        self.testPredict = scaler.inverse_transform(self.testPredict).reshape(-1, 1)  
        testY = scaler.inverse_transform(self.testY).reshape(-1, 1)
        self.unseen_label = scaler.inverse_transform(self.unseen_label).reshape(-1, 1)
        #plot_expected_vs_actual(self.testPredict, self.unseen_label)
        

    def create_dataset(self, dataset, look_back_val):
        '''
        Create a dataset of the look_back_val

        :param dataset: the dataset to be used
        :param look_back_val: the look_back_val

        :return: the datasets X and Y of the look_back_val
        '''
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back_val - 1):
            dataX.append(dataset[i:(i + look_back_val)])
            dataY.append(dataset[i + look_back_val])
        return np.array(dataX), np.array(dataY)

    def validate(self):
        '''
        Using TimeSeriesSplit, we validate the model
        '''
        tscv = TimeSeriesSplit(n_splits=5)
        for train_index, test_index in tscv.split(self.trainX):
            print("TRAIN:", train_index, "TEST:", test_index)
            self.trainX, self.trainY = self.trainX[train_index], self.trainY[train_index]
            self.testX, self.testY = self.testX[test_index], self.testY[test_index]
            model = Sequential()
            model.add(LSTM(4, input_shape=(1, self.look_back_val), return_sequences=False))
            model.add(Dense(1))
            model.compile(loss='mean_squared_error', optimizer='adam')
            model.fit(self.trainX, self.trainY, epochs=100, batch_size=1, verbose=2)
            self.trainPredict = model.predict(self.trainX)
            self.testPredict = model.predict(self.testX)
            self.trainPredict = scaler.inverse_transform(self.trainPredict).reshape(-1, 1)
            trainY = scaler.inverse_transform(self.trainY).reshape(-1, 1)
            self.testPredict = scaler.inverse_transform(self.testPredict).reshape(-1, 1)  
            testY = scaler.inverse_transform(self.testY).reshape(-1, 1)
            self.unseen_label = scaler.inverse_transform(self.unseen_label).reshape(-1, 1)
            plot_expected_vs_actual(self.testPredict, self.unseen_label)
            print("Mean squared error: %.2f" % mean_squared_error(testY, self.testPredict))
            print('\n')