from sklearn.linear_model import LogisticRegression
from preprocess import csvToDataFrame, train_test_split, trip_total_to_int, int_to_label, preprocess
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from viz import plot_expected_vs_actual, plot_data_distribution



#------------
#|Model|Accuracy|num_classes|
#|regression (pre-standardization)|0.56|6|
#|regression (post-standardization)|0.575|6|
#|regression (post-standardization)|0.41|12|
#|regression (post-standardization)|0.28|28(for each price 13(min)-33(max))|



class regressionModel(object):

    '''Our baseline regression Model
     We use the one-vs-one approach to train a binary classifier for each pair of labels
     We normalize the data because in multivariate linear regression, the attribute with different scales will affect the result
     even if it is not important to the predictor at all.

     From PCA analysis, we learned that datapoints with different classes overlap each other. Hence, with this model, we performed 
     pretty badly.
     '''

    def __init__(self, data, unseen_data):
       
        # Each dataset has below features
        # trip_total,trip_pooled,trip_miles,cases_total,deaths_total,hospitalizations_total,cases_age_18_29,cases_age_60_69,year,month

        self.train_data, self.test_data, self.train_label, self.test_label, self.unseen_data, self.unseen_label, self.df, self.df_label = preprocess(data, unseen_data)
        self.pipe = make_pipeline(StandardScaler(), LogisticRegression(solver='lbfgs', multi_class='auto'))
        self.train(self.train_data, self.train_label)
        # acc = self.score(self.test_data, self.test_label)
        # print("Accuracy: ", acc)

        self.predictions = self.predict(self.unseen_data)
        
        # Plot the results
        #plot_expected_vs_actual(self.predictions, self.unseen_label)
        #plot_data_distribution(df)

    def train(self, train_data, train_label, pipe = True):
        '''
        Train the model

        :param train_data: the training data
        :param train_label: the training labels
        :param pipe: whether to use the pipeline or not (default True)
        '''
        if pipe:
            self.pipe.fit(train_data, train_label)
        else:
            self.model.fit(train_data, train_label)

    def score(self, test_data, test_label, pipe = True):
        '''
        Score the model

        :param test_data: the test data
        :param test_label: the test labels
        :param pipe: whether to use the pipeline or not (default True)

        :return: the accuracy of the model
        '''
        if pipe:
            return self.pipe.score(test_data, test_label)
        else:
            return self.model.score(test_data, test_label)
    
    def predict(self, unseen_data, pipe = True):
        '''
        Based on the input data, predict their outcomes and return them

        :param unseen_data: the unseen data
        :param pipe: whether to use the pipeline or not (default True)

        :return: the predicted outcomes
        '''

        if pipe:
            return self.pipe.predict(unseen_data)
        else:
            return self.model.predict(unseen_data)

