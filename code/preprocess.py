import numpy as np
import pandas as pd
import csv
import random
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler


# Set the random seed for reproducibility
RANDOM_SEED = 0
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def csvToDataFrame(fileName):
    '''
    Read in the csv file and return a pandas dataframe object

    :param fileName: the path to the csv file

    :return: the dataframe object
    '''
    df = pd.read_csv(fileName)
    return df

def returnYear(x):
    '''
    Helper function for converting timestamp to corresponding year. We assume x to be in conventional timestamp format 

    :param x: the timestamp
    
    :return: the corresponding year
    '''
    if int(x[:4]) == 2020:
        return 2020 
    elif int(x[:4]) == 2021:
        return 2021
    elif int(x[:4]) == 2022:
        return 2022

def join():
    '''
    # This functions joins the Covid Table and Transportation Table. The two tables outer join on timestamp. 
    # The joint table is then written to join.csv for easier access later on
    '''

    # dictionary for month
    month_dict = {1: "jan", 2: "feb", 3: "mar", 4: "apr", 5: "may", 6: "jun", 7: "jul", 8: "aug", 9: "sep", 10: "oct", 11: "nov", 12: "dec"}
    for i in range(2020, 2023, 1):
        for j in range(1, 13):
            if i == 2022 and j == 3: #This is because our Covid data ends in Feb of 2022
                break
            if i == 2020 and j == 1:
                #Reads in each month file for specific year and append to make a whole df capturing all years
                trans_df = csvToDataFrame("../data_deliverable/data/"+str(i)+"/" + month_dict[j] + ".csv") 
            else:
                trans_df = trans_df.append(csvToDataFrame("../data_deliverable/data/"+str(i)+"/" + month_dict[j] + ".csv"))
    
    
    covid_df = csvToDataFrame("../data_deliverable/data/covid/covid_whole.csv")
    #Merge the two tables on timestamp
    joined_df = pd.merge(trans_df, covid_df, on='trip_start_timestamp', how = 'outer')
    #Replace the nan with 0
    joined_df = joined_df.fillna(0)
    #Add new columns to the joint table for later uses
    joined_df['year'] = joined_df['trip_start_timestamp'].map(lambda x: returnYear(x))
    joined_df['month'] = joined_df['trip_start_timestamp'].map(lambda x: int(x[5:7]))
    #Write to the join.csv file
    joined_df.to_csv("../data_deliverable/data/join.csv", sep=",", index=False)


def train_test_split(filename, train_pct=0.8):
    '''
    Split the data into train and test data based on the percentage (defaulted to 4:1)

    :param filename: the path to the csv file
    :param train_pct: the percentage of the data to be used for training

    :return: train_df, test_df, df (the raw data)
    '''

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    df = csvToDataFrame(filename)
    msk = np.random.rand(len(df)) < train_pct

    # We drop these attributes because they are irrelevant and might unfairly bias the model
    df = df.drop("trip_start_timestamp", axis=1)
    df = df.drop("year", axis=1)
    df = df.drop("month", axis=1)
    return df[msk], df[~msk], df


def trip_total_to_int(x):
    '''
    Function that rounds up the float input

    :param x: the float input

    :return: the rounded up float
    '''
    return round(x)

def process_unseen_data(filename):
    '''
    Process the unseen data and return the data and the label. Unseen data is used to predict the "future price" 
    "for which we know the price of"

    :param filename: the path to the csv file

    :return: unseen_data, unseen_label
    '''

    df = csvToDataFrame(filename)
    df = df.drop("trip_start_timestamp", axis=1)
    df = df.drop("year", axis=1)
    df = df.drop("month", axis=1)
    df_label = df.pop('trip_total').apply(lambda x: trip_total_to_int(x)).apply(lambda x: int_to_label(x)).to_numpy()
    scalar = MinMaxScaler()
    df = scalar.fit_transform(df)
    return df, df_label

def int_to_label(x):
    '''
    Classify x to its appropriate label
    TODO: how many labels do we want for our dataset? (How accurate do we want our model to be?)

    :param x: the trip_total we want to convert to a label

    :return: the label for the trip_total

    '''
    if x == 13:
        return 13
    elif x == 14:
        return 14
    elif x == 15:
        return 15
    elif x == 16:
        return 16
    elif x == 17:
        return 17
    elif x == 18:
        return 18
    elif x == 19:
        return 19
    elif x == 20:
        return 20
    elif x == 21:
        return 21
    elif x == 22:
        return 22
    elif x == 23:
        return 23
    elif x == 24:
        return 24
    elif x == 25:
        return 25
    elif x == 26:
        return 26
    elif x == 27:
        return 27
    elif x == 28:
        return 28
    elif x == 29:
        return 29
    elif x == 30:
        return 30
    elif x == 31:
        return 31
    elif x == 32:
        return 32
    elif x == 33:
        return 33
    elif x == 34:
        return 34
    elif x == 35:
        return 35

    # if x in range(13, 16):
    #     return 0
    # elif x in range(16, 19):
    #     return 1
    # elif x in range(19, 22):
    #     return 2
    # elif x in range(22, 25):
    #     return 3
    # elif x in range(25, 28):
    #     return 4
    # elif x in range(28, 31):
    #     return 5
    # elif x in range(31, 34):
    #     return 6


    # if x in range(13, 15):
    #     return 0
    # elif x in range(15, 17):
    #     return 1
    # elif x in range(17, 19):
    #     return 2
    # elif x in range(19, 21):
    #     return 3
    # elif x in range(21, 23):
    #     return 4
    # elif x in range(23, 25):
    #     return 5
    # elif x in range(25, 27):
    #     return 6
    # elif x in range(27, 29):
    #     return 7
    # elif x in range(29, 31):
    #     return 8
    # elif x in range(31, 33):
    #     return 9
    # elif x in range(33, 35):
    #     return 10

def preprocess(filename, filename_unseen):
    '''
    preprocess the data and return the final train, test and unseen data with their labels as well as the raw DataFrame

    :param filename: the path to the csv file
    :param filename_unseen: the path to the csv file for unseen data

    :return: train_data, test_data, unseen_data, train_label, test_label, unseen_label, df

    '''
    train_df, test_df, df = train_test_split(filename)
    df_unseen, df_unseen_label = process_unseen_data(filename_unseen)
    

    # Min price is $14.06, Max price is $32.58
    train_label = train_df.pop('trip_total').apply(lambda x: trip_total_to_int(x)).apply(lambda x: int_to_label(x)).to_numpy()

    # Min price is $13.94, Max price is $32.36
    test_label = test_df.pop('trip_total').apply(lambda x: trip_total_to_int(x)).apply(lambda x: int_to_label(x)).to_numpy()

    # scale the data
    scalar = MinMaxScaler(feature_range=(0, 1))
    train_df = scalar.fit_transform(train_df)
    test_df = scalar.transform(test_df)

    df_label = df.pop('trip_total').apply(lambda x: trip_total_to_int(x)).apply(lambda x: int_to_label(x)).to_numpy()

    df = scalar.transform(df)

    return train_df, test_df, train_label, test_label, df_unseen, df_unseen_label, df, df_label

def label_to_plot(to_convert):
    '''
    Convert the label to x and y coordinate for plotting
    '''

    x = []
    y = []

    for i in range(len(to_convert)):
        x.append(i+1) # day of the month
        y.append(to_convert[i])

    return x, y