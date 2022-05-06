from pandas import read_csv
from statsmodels.tsa.stattools import adfuller
from scipy.stats import chi2_contingency, chi2
import pandas as pd
from pingouin import multivariate_normality


def hypothesis1():
    '''
    Null Hypothesis: Covid cases and Trip price are independent of each other.

    Alternative Hypothesis: Covid cases and Trip price are dependent of each other.

    Motivation: We want to learn if there is any relationship between the two attributes (`trip_total` and `cases_total`)

    P-Value: 0.05 (0.05 is the significance level)

    Test Used: Chi-Square Test 

    Result: Since p < 0.05, we reject the null hypothesis. The two attributes in question are dependent of each other.

    --------------------------------------------------------------------------------
    Chi-Square Statistic: 10705.839180
    p-value: 0.0001 
    Critical Value: 10323.781777
    --------------------------------------------------------------------------------

    '''
    
    # Make data categorical
    series = read_csv('../data_deliverable/data/join.csv', header=0, usecols=['cases_total', 'trip_total'], engine='python')
    series = series.dropna()
    series = series.astype('int')

    # Cross Tabulaton
    table = pd.crosstab(series['cases_total'], series['trip_total'])
    stat, p, dof, expected = chi2_contingency(table)
    print('Statistic: %f' % stat)
    print('p-value: %f' % p)

    # Alternative way of checking if we want to reject the null hypothesis
    prob = 0.95
    critical = chi2.ppf(prob, dof)
    print('Critical Value: %f' % critical)

    # Since this if condition is also true, we reject the null hypothesis.
    if abs(stat) > critical: 
        print('Probably significant with %f' % prob)
    
    


def hypothesis2():
    '''
    Null Hypothesis: The time series data for `trip_total` we have is non stationary, in other words, it has a unit root. 
    It is much affected by the Covid-19 pandemic.

    Alternative Hypothesis: The time series data for `trip_total` we have is stationary, in other words, it does not
    have time-depenedent structure.

    Motivation: By learning this additional information, we might be able to do feature engineering
    and feature selection to make our Machine Learning model fit better and perform well.

    P-Value: 0.05 (0.05 is the significance level)

    Test Used: Augmented Dickey-Fuller Test

    Result: Since p > 0.05, we fail to reject the null hypothesis. Therefore, our time series data is non-stationary.
    --------------------------------------------------------------------------------
    ADF Statistic: -1.349072
    p-value: 0.606434
    Critical Values:
        1%  : -3.439
        5%  : -2.865
        10% : -2.569
    --------------------------------------------------------------------------------

    '''
    
    series = read_csv('../data_deliverable/data/join.csv', header=0, index_col = 0, squeeze=True)

    # Extract the series
    series = series['trip_total']

    result = adfuller(series.values)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))



def hypothesis3():
    '''
    Null Hypothesis: Our time series data set has a Gauassian distribution.

    Alternative Hypothesis: Our time series data set has a non-Gauassian distribution.

    Motivation: We want to learn if our time series data set has a Gaussian distribution or not because
    Machine Learning model makes the assumption that any input dataset has this property. From learning this,
    we know if we will need to preprocess the data so that they will then have Gaussian distribution.

    P-Value: 0.05 (0.05 is the significance level)

    Test Used: Multivariate Normality Test

    Result: Since p < 0.05, we reject the null hypothesis. Therefore, our time series data set is not MVN.

    --------------------------------------------------------------------------------
    p-value: 0.0012
    '''
    
    # Read and drop irrelevant columns
    series = read_csv('../data_deliverable/data/join.csv', header=0, index_col = 0, squeeze=True)
    series = series.drop(['year', 'month'], axis=1)
    
    # Apply anderson-darling test
    hz, p, normal = multivariate_normality(series, alpha = 0.05)
    print(p, normal)
    

