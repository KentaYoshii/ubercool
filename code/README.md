# How to Run
Here are the flags you can use to run the program:
```
# cd to the directory where the main.py is located

python main.py [--help] [--k K] [--model MODEL] [--load LOAD]
               [--plot PLOT] [--compare COMPARE] [--hypotheses HYPOTHESES]

optional arguments:
    --help, -h                                 Show this help message and exit
    --k K                                      Number of clusters to use
    --model MODEL                              Model to use 
                                                (regression(default), kmeans, or lstm)
                                                         
    --load LOAD                                Load dataset using api (True or False)
    --plot PLOT                                Plot the results (True or False)
    --compare COMPARE                          Train all models and compare by plotting (True or False)
    --hypotheses HYPOTHESES                    Run the hypotheses tests (True or False)
``` 

# Design Choices
Our program can be divided into three parts:
1. Data loading 
   - This part of the program loads the data from the api of Chicago's open data portal. Since the data set is too large (35,000 entries for each day), we decided to take the average of 35,000 entries per day so that our finalized data set is smaller. All of this part is defined in the dataload.py.
2. Pre-processing
   - In this part, we do the data preprocessing before passing it to train and test the model. Here we do the following:
     - convert column values to discrete values (e.g. "Yes" to 1, 13.4 to 13, etc.)
     - join the covid data set with the transport dataset
     - split the dataset into train and test dataset
     - Finally we apply scaling to the data so that no one feature is dominating the others. (which might affect models' decision-making)
3. Model training
   - Here we have three models: liner regression, KMeans + SVM and LSTM. You can specify which model you want to use by using the --model flag. We used sklearn's LinearRegression, KMeans and SVM models respectively. For LSTM, we used tensorflow's LSTM model. The model will fit on the train dataset and then predict the values on the test dataset. To compute the final performance (which we measure using MSE), we used forward chaining and find the MSE at each stage of the model. Then we take the average of all the MSEs. 
4. Data visualization
   - Finally, we plot the results of the model's predictions against the ground truth price labels. We used the matplotlib's seaborn library to plot the results. 
# Note:
To load the dataset from scratch, you will need to go to dataload.py and add the TOKEN you create from socrata.