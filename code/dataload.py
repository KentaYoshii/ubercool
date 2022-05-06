from socket import timeout
from unittest import result
from sodapy import Socrata
import pandas as pd
import requests
import csv
import time
import numpy as np

#https://dev.socrata.com/docs/app-tokens.html 
#To prevent from throttle to happen easily

# TODO: create your own account from the link above, and fill in the three variables
# below with your own account information.

APP_TOKEN = None
USERNAME = None
PASSWORD = None

HOME_PATH = "data.cityofchicago.org"
COVID_API_ENDPOINT = "naz8-j4nc"
TRANS_ENDPOINT = "m6dm-c72p"

def data_load_Covid():
    '''
    Load the Covid data from an endpoint and write to a new file.
    '''
    PATH_TO_COVID = open("../data_deliverable/data/covid/covid_whole.csv", "a")
    client = Socrata(HOME_PATH, 
                     APP_TOKEN,
                     USERNAME,
                     PASSWORD)

    # Create a query
    select = "lab_report_date, cases_total, deaths_total, hospitalizations_total, cases_age_18_29, cases_age_60_69" 
    order= "lab_report_date ASC"

    resultsCov = client.get(COVID_API_ENDPOINT, select=select, order=order)
    # Create a dataframe
    results_df_cov = pd.DataFrame.from_records(resultsCov)
    for ind in results_df_cov.index:
        covid_dict = {"lab_report_date": results_df_cov["lab_report_date"][ind],"cases_total": results_df_cov["cases_total"][ind], "deaths_total": results_df_cov["deaths_total"][ind], "hospitalizations_total": results_df_cov["hospitalizations_total"][ind], "cases_age_18_29": results_df_cov["cases_age_18_29"][ind], "cases_age_60_69": results_df_cov["cases_age_60_69"][ind]}
        writer = csv.writer(PATH_TO_COVID)
        
    PATH_TO_COVID.close()

def data_load_Trans():
    '''
    Load the Tranport data from an endpoint and write to a new file. 
    Since it is a huge data, we do it manually by month to prevent from throttling from happening.
    '''

    print("data load starting: ")
    # Open the csv file we want to write to 
    PATH_JAN_FILE = open("../data_deliverable/data/2022/feb.csv", "a")

    # Create our client with the Key and other appropriate params
    client = Socrata(HOME_PATH,
                     APP_TOKEN, 
                     USERNAME,
                     PASSWORD,
                     timeout=10000)

    for i in range(1, 29):
        print("data load: " + str(i))
        # Create a query
        if i < 10:
            where = "trip_start_timestamp between '2022-02-0" + str(i) +"T00:00:00.000' and '2022-02-0" + str(i) +'T23:59:59.999'"'"
        else:
            where = "trip_start_timestamp between '2022-02-" + str(i) +"T00:00:00.000' and '2022-02-" + str(i) +'T23:59:59.999'"'"
        select = "avg(trip_total) as trip_total, avg(trips_pooled) as trip_pooled, avg(trip_miles) as trip_miles"
        group = "trip_start_timestamp"

        try:
            # Query the API
            resultsTrans = client.get(TRANS_ENDPOINT,select=select, where=where, group=group)
        except requests.exceptions.Timeout:
            print("Timeout Occured")

        # crate df and take the mean of the columns
        results_df_trans = pd.DataFrame.from_records(resultsTrans)  
        attToAvg = {"trip_start_timestamp": "2022-02-"+str(i), "trip_total": results_df_trans["trip_total"].astype(float).mean(), "trip_pooled": results_df_trans["trip_pooled"].astype(float).mean(), "trip_miles": results_df_trans["trip_miles"].astype(float).mean()}
        # write
        writer = csv.writer(PATH_JAN_FILE)
        writer.writerow(attToAvg.values())
    PATH_JAN_FILE.close()
 
def take_average(PATH_R, PATH_W, monthly=True):
    '''
    A function that takes the average of each column value for month specified in PATH_R
    This is for the Transport data

    :param PATH_R: file we want to read from
    :param PATH_W: file we want to write to
    :param monthly: if true, we will take the average of each month
    '''

    # Temp storage for each day in a month
    trip_total_arr = np.zeros(31)
    trip_pooled_arr = np.zeros(31)  
    trip_miles_arr = np.zeros(31)

    count = 0
    if monthly:
        with open(PATH_R, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # for each row in PATH_R, we store the value in the corresponding array
                month = row["trip_start_timestamp"][:7]
                trip_total_arr[count] += float(row["trip_total"])
                trip_pooled_arr[count] += float(row["trip_pooled"])
                trip_miles_arr[count] += float(row["trip_miles"])
                count += 1

            # Take the mean of each array
            avg_trip_total = np.sum(trip_total_arr) / count
            avg_trip_pooled = np.sum(trip_pooled_arr) / count
            avg_trip_miles = np.sum(trip_miles_arr) / count

            # Write to file
            avg_dict = {"date": month,"avg_trip_total": avg_trip_total, "avg_trip_pooled": avg_trip_pooled, "avg_trip_miles": avg_trip_miles}
            PATH_W = open(PATH_W, "a")
            writer = csv.writer(PATH_W)
            writer.writerow(avg_dict.values())
            PATH_W.close()

def take_average_covid(PATH_R, PATH_W, monthly=True):
    '''
    Given a csv file (PATH_R), we will take the average of each month and write to a file (PATH_W)
    This is for the COVID data

    :param PATH_R: file we want to read from
    :param PATH_W: file we want to write to
    :param monthly: if true, we will take the average of each month

    :return: None
    '''

    # Initialize our dicts
    month_dict = {"01": [], "02": [], "03": [], "04": [], "05": [], "06": [], "07": [], "08": [], "09": [], "10": [], "11": [], "12": []}
    month_toExtract = {"01", "02"}
    month_toExtract2 = {"03", "04", "05", "06", "07", "08", "09", "10", "11", "12"}
    month_subset = {key: month_dict[key] for key in month_toExtract}
    month_subset2 = {key: month_dict[key] for key in month_toExtract2}
    year_dict = {"2020": month_subset2, "2021": month_dict, "2022": month_subset}

    with open(PATH_R, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cur_year = row["lab_report_date"][:4] # Get the year
            month = row["lab_report_date"][5:7] # Get the month
            year_dict[cur_year][month].append(row)
    PATH_W = open(PATH_W, "a")
    writer = csv.writer(PATH_W)

    # For each year, we will take the average of each month, one attribute at a time
    for year in year_dict.keys():
        for month in year_dict[year].keys():
            curList = year_dict[year][month]
            cases_total_arr = np.zeros(len(curList))
            deaths_total_arr = np.zeros(len(curList))
            hospitalizations_total_arr = np.zeros(len(curList))
            cases_age_18_29_arr = np.zeros(len(curList))
            cases_age_60_69_arr = np.zeros(len(curList))
            for idx, val in enumerate(curList):
                cases_total_arr[idx] += float(val["cases_total"])
                deaths_total_arr[idx] += float(val["deaths_total"])
                hospitalizations_total_arr[idx] += float(val["hospitalizations_total"])
                cases_age_18_29_arr[idx] += float(val["cases_age_18_29"])
                cases_age_60_69_arr[idx] += float(val["cases_age_60_69"])
            avg_cases_total = np.sum(cases_total_arr) / len(curList)
            avg_deaths_total = np.sum(deaths_total_arr) / len(curList)
            avg_hospitalizations_total = np.sum(hospitalizations_total_arr) / len(curList)
            avg_cases_age_18_29 = np.sum(cases_age_18_29_arr) / len(curList)
            avg_cases_age_60_69 = np.sum(cases_age_60_69_arr) / len(curList)
            avg_dict = {"date": year+"-"+month, "avg_cases_total": avg_cases_total, "avg_deaths_total": avg_deaths_total, "avg_hospitalizations_total": avg_hospitalizations_total, "avg_cases_age_18_29": avg_cases_age_18_29, "avg_cases_age_60_69": avg_cases_age_60_69}
            writer.writerow(avg_dict.values())

    PATH_W.close()