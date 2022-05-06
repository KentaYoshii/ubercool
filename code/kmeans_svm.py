from sklearn.linear_model import LogisticRegression
from preprocess import csvToDataFrame, train_test_split, trip_total_to_int, int_to_label, preprocess
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import warnings
from sklearn.svm import SVR
from viz import plot_data_distribution, plot_expected_vs_actual
from preprocess import trip_total_to_int, int_to_label
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings('ignore')


class Kmeans_SVM(object):

    def __init__(self, filename, filename_unseen, num_clusters = 3):
        self.train_data, self.test_data, self.train_label, self.test_label, self.unseen_data, self.unseen_label, self.df, self.df_label = preprocess(filename, filename_unseen)
       
        #Uncomment this when you do elbow plot
        #self.df = self.df.drop(['trip_total'], axis=1)
        
        self.num_clusters = num_clusters
        #self.find_best_cluster_num()

        self.model = KMeans(n_clusters=self.num_clusters)
        self.train_kmeans()
        self.prediction = self.predict_kmeans()
       
        self.cluster_center = self.get_cluster_center()

        #print(self.train_data.shape) # sample size by 7 features
  
        # Make a new DataFrame with new cluster and label columns
        train_df = pd.DataFrame(self.train_data, columns=['trip_pooled', 'trip_miles', 'cases_total', 'deaths_total', 'hospitalizations_total', 'cases_age_18_29', 'cases_age_60_69'])
        train_df.insert(7, "trip_total", self.train_label)
        train_df.insert(8, "cluster", self.prediction)
        train_cluster_df = []

        # Create a list of dataframes for each cluster
        for i in range(self.num_clusters):
            train_cluster_df.append(train_df[train_df['cluster'] == i])

        # Create a list of SVR models for each cluster
        cluster_svr = []
        model = SVR(kernel='rbf', C=1e3, epsilon = 1)

        for i in range(self.num_clusters):
            cluster_X = np.array((train_cluster_df[i])[['trip_pooled', 'trip_miles', 'cases_total', 'deaths_total', 'hospitalizations_total', 'cases_age_18_29', 'cases_age_60_69']])
            cluster_Y = np.array((train_cluster_df[i])['trip_total'])
            cluster_svr.append(model.fit(cluster_X, cluster_Y))

        # Predict the trip_total for unseen data
        self.Y_svr_k_means_pred, Y_clusters = self.regression_function(self.unseen_data, self.model, cluster_svr)

        # Make a new result DataFrame with new cluster and predicted label columns
        result_df = pd.DataFrame(self.unseen_data, columns=['trip_pooled', 'trip_miles', 'cases_total', 'deaths_total', 'hospitalizations_total', 'cases_age_18_29', 'cases_age_60_69'])
        result_df["trip_total_true"] = self.unseen_label
        result_df["trip_total_pred"] = self.Y_svr_k_means_pred
        result_df["cluster"] = Y_clusters
        
        # Plot the distribution of trip_total for each cluster
        #plot_expected_vs_actual(result_df['trip_total_pred'], result_df['trip_total_true'])
        self.predictions = result_df['trip_total_pred']

    def regression_function(self, arr, kmeans, cluster_svr):
        '''
        Apply regression function to predict the trip_total for unseen data

        :param arr: unseen data
        :param kmeans: kmeans model
        :param cluster_svr: list of SVR models for each cluster

        :return: predicted trip_total for unseen data
        :return: predicted cluster for unseen data
        '''
        result = []
        clusters_pred = kmeans.predict(arr)
        for i, data in enumerate(arr):
            result.append(((cluster_svr[clusters_pred[i]]).predict([data]))[0])
        return result, clusters_pred


    def train_kmeans(self):
        '''
        Fit the model
        '''
        self.model.fit(self.train_data)

    def predict_kmeans(self):
        '''
        Predict the label for unseen data
        '''
        return self.model.predict(self.train_data)

    def score_kmeans(self):
        '''
        Calculate the score of the model
        '''

        return accuracy_score(self.unseen_label, self.Y_svr_k_means_pred)

    def get_cluster_center(self):
        '''
        Get the cluster center for each cluster

        :return: cluster_center
        '''
        return self.model.cluster_centers_


    def find_best_cluster_num(self):
        '''
        Using elbow curve method, we find the best number of clusters for this 
        particular dataset. (We will use 3 clusters)
        '''
        list_to_check = [x for x in range(1, 11)]
        kmeans = [KMeans(n_clusters=i) for i in list_to_check]
        score = [kmeans[i].fit(self.df.to_numpy()).score(self.df.to_numpy()) for i in range(len(kmeans))]
        score
        plt.plot(list_to_check,score)
        plt.xlabel('Number of Clusters')
        plt.ylabel('Score')
        plt.title('Elbow Curve')
        plt.show()

    def validate(self):
        '''
        Validate the model using TimeSeriesSplit. We use this because we have a timeseries
        dataset.
        '''
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        for train_index, test_index in tscv.split(self.df):
            X_train, X_test = self.df.iloc[train_index], self.df.iloc[test_index]
            y_train, y_test = self.df_label.iloc[train_index], self.df_label.iloc[test_index]
            model = SVR(kernel='rbf', C=1e3, epsilon = 1)
            model.fit(X_train, y_train)
            scores.append(model.score(X_test, y_test))