import argparse
import os
from dataload import data_load_Covid, data_load_Trans, take_average, take_average_covid
from preprocess import join, train_test_split
from regressionModel import regressionModel
from kmeans_svm import Kmeans_SVM
from lstm import Lstm
from viz import plot_result, plot_expected_vs_actual
from sklearn.metrics import mean_squared_error

from hypotheses import hypothesis1, hypothesis2, hypothesis3


PATH = "../data_deliverable/data/join.csv"
PATH2 = "../data_deliverable/data/join_test.csv"

def parseArguments():
    '''
    Parse the command line arguments of the program.
    '''
    parser = argparse.ArgumentParser(description = "A simple script to load a data with specific attributes")
    parser.add_argument("--limit", type=int, default=150, help="limit the number of rows")
    parser.add_argument("--model", type=str, default="regression", help="model to run")
    parser.add_argument("--k", type=int, default=3, help="number of clusters")
    parser.add_argument("--load", type=str, default= False, help = "boolean for loading data")
    parser.add_argument("--plot", type=str, default= False, help = "boolean for plotting")
    parser.add_argument("--compare", type=str, default= False, help = "train all models and compare results")
    parser.add_argument("--hypotheses", type=str, default= False, help = "run the hypothesis")
    args = parser.parse_args()

    return args

def main(args):
    '''
    Run releveant function here 

    :param args: the arguments of the program
    '''

    # Load Data here
    if args.load == "True":
        Covid_df = data_load_Covid()
        Trans_df = data_load_Trans()
        #take_average("../data_deliverable/data/2022/jan.csv", "../data_deliverable/data/average/monthly_average.csv")
        #take_average_covid("../data_deliverable/data/covid/covid_whole.csv", "../data_deliverable/data/covid/covid_monthly_average.csv")
        join()

    # Instantiate our three models here as needed
    if args.model == "regression":
        reg = regressionModel(PATH, PATH2)
        print(mean_squared_error(reg.unseen_label, reg.predictions), "regression")
        if args.plot == "True":
            plot_expected_vs_actual(reg.predictions, reg.unseen_label)
    elif args.model == "kmeans":
        kmeans = Kmeans_SVM(PATH, PATH2,  args.k)
        print(mean_squared_error(kmeans.unseen_label, kmeans.predictions), "kmeans")
        if args.plot == "True":
            plot_expected_vs_actual(kmeans.predictions, kmeans.unseen_label)
    elif args.model == "lstm":
        lstm = Lstm(PATH, PATH2)
        print(mean_squared_error(lstm.unseen_label, lstm.predictions), "lstm")
        if args.plot == "True":
            plot_expected_vs_actual(lstm.predictions, lstm.unseen_label)
    elif args.compare == "True":
        reg = regressionModel(PATH, PATH2)
        kmeans = Kmeans_SVM(PATH, PATH2,  args.k)
        lstm = Lstm(PATH, PATH2)
        print(mean_squared_error(reg.unseen_label, reg.predictions), "regression")
        print(mean_squared_error(kmeans.unseen_label, kmeans.predictions), "kmeans")
        print(mean_squared_error(lstm.unseen_label, lstm.predictions), "lstm")
        plot_result(reg.predictions, kmeans.predictions, lstm.testPredict, lstm.unseen_label)


    # Call the hypothesis functions here
    if args.hypothesis == "True":
        hypothesis1()
        hypothesis2()
        hypothesis3()


if __name__ == "__main__":
    args = parseArguments()
    main(args)