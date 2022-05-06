# __Data__ 
### We have two datasets that we will use to train our model.
- [Chicago COVID-19 Data](https://data.cityofchicago.org/Health-Human-Services/COVID-19-Daily-Cases-Deaths-and-Hospitalizations/naz8-j4nc)
- [Chicago Transportation Network Providers](https://data.cityofchicago.org/Transportation/Transportation-Network-Providers-Trips/m6dm-c72p)

## __Data Description__
```javascript
    { // Json data for the original Chicago COVID-19 Data

        // This represents a single data point
        'lab_report_date': '2022-02-28T00:00:00.000',
        'cases_total': '2',
        'deaths_total': '0',
        `hospitalizations_total`: "0",    
        'cases_age_0_17': '0',
        'cases_age_18_29': '0',
        'cases_age_30_39': '1',
        'cases_age_40_49': '0',
        'cases_age_50_59': '0',
        'cases_age_60_69': '0',
        'cases_age_70_79': '0',
        'cases_age_80_': '1',
        'cases_age_unknown': '0',
        'cases_female': '0',
        'cases_male': '2',
        'cases_unknown_gender': '0',
        'cases_latinx': '0',
        'cases_asian_non_latinx': '0',
        'cases_black_non_latinx': '0', 
        'cases_white_non_latinx': '2', 
        'cases_other_non_latinx': '0', 
        'cases_unknown_race_eth': '0', 
        'deaths_0_17_yrs': '0',
        'deaths_18_29_yrs': '0',
        'deaths_30_39_yrs': '0',
        'deaths_40_49_yrs': '0', 
        'deaths_50_59_yrs': '0', 
        'deaths_60_69_yrs': '0', 
        'deaths_70_79_yrs': '0',
        'deaths_80_yrs': '0', 
        'deaths_unknown_age': '0',
        'deaths_female': '0',
        'deaths_male': '0',
        'deaths_unknown_gender': '0',
        'deaths_latinx': '0', 
        'deaths_asian_non_latinx': '0',
        'deaths_black_non_latinx': '0', 
        'deaths_white_non_latinx': '0', 
        'deaths_other_non_latinx': '0', 
        'deaths_unknown_race_eth': '0'

        }
```
## The Keys we are interested in Extracting
- `lab_report_date`: report date
  - Type: floating_timestamp
  - Default Value: None
  - Range of Value: 03/01/2020-02/28/2022
  - Distribution of values: Uniform 
  - This is an identifier
  - These values are unique
  - This value is unique so it will not be used to detect duplicate data point
  - It is a required value
  - Yes. We will use this value to join with the transportation table to analyze the relationship between covid-19 and ride-sharing.
  - No, it does not include any sensitive information
- `cases_total`: total number of cases on that day
  - Type: number
  - Default Value: None
  - Range of Value: 0 - 10197
  - Distribution of values: Normal 
  - This is not an identifier
  - These values are not unique
  - This value will not be used to detect duplicate data point
  - It is a required value
  - Yes. We will use this value to capture the trend of covid-19 and how that might affect ride-sharing
  - No, it does not include any sensitive information
- `deaths_total`: total number of deaths on that day
  - Type: number
  - Default Value: None
  - Range of Value: 0 - 58
  - Distribution of values: Normal 
  - This is not an identifier
  - This value will not be used to detect duplicate data point
  - It is a required value
  - Yes. We will use this value to capture the trend of covid-19 and how that might affect ride-sharing
  - No, it does not include any sensitive information
- `hospitalizations_total`: total number of newly hospitalized people on that day
  - Type: number
  - Default Value: None
  - Range of Value: 1 - 684
  - Distribution of values: Roughly Normal
  - This is not an identifier
  - These values are not unique
  - This value will not be used to detect duplicate data point
  - It is an optional value
  - No, it does not include any sensitive information

- `cases_age_60_69`: tip people paid
  - Type: number
  - Default Value: 0
  - Range of Value: 0 - 775
  - Distribution of values: Uniform 
  - This is not an identifier
  - This value will not be used to detect duplicate data point
  - It is an optional value
  - Yes. We will use this value to analyze if certain age group use more transportation
  - No, it does not include any sensitive information
  
- `cases_age_18_29`: age range of covid cases
  - Type: number
  - Default Value: 0
  - Range of Value: 0 - 2216
  - Distribution of values: Uniform 
  - This is not an identifier
  - This value will not be used to detect duplicate data point
  - It is an optional value
  - Yes. We will use this value to analyze if certain age group use more transportation 
  - No, it does not include any sensitive information


```javascript
        { // Json data from the original Chicago Transportation Providers

            // This represents a single trip
            'trip_id': 'ae90ffd77e56a7f1334e2af0a6d0bae725b52ea1', 
            'trip_start_timestamp': '2018-11-17T19:30:00.000', 
            'trip_end_timestamp': '2018-11-17T19:45:00.000',
            'trip_seconds': '1024',
            'trip_miles': '3.19024462550898',
            'pickup_census_tract': '17031070500',
            'dropoff_census_tract': '17031830800',
            'pickup_community_area': '7',
            'dropoff_community_area': '4',
            'fare': '10',
            'tip': '5',
            'additional_charges': '2.5',
            'trip_total': '17.5',
            'shared_trip_authorized': False,
            'trips_pooled': '1',
            'pickup_centroid_latitude': '41.9289459041', 
            'pickup_centroid_longitude': '-87.6608925701', 
            'pickup_centroid_location': {'type': 'Point', 
            'coordinates': [-87.6608925701, 41.9289459041]}, 
            'dropoff_centroid_latitude': '41.9651417087', 
            'dropoff_centroid_longitude': '-87.6765780714', 
            'dropoff_centroid_location': {'type': 'Point', 
            'coordinates': [-87.6765780714, 41.9651417087]
        }
```
## The Keys we are interested in Extracting
- `trip_id`: identifier for a trip
  - Type: text
  - Default Value: None
  - Range of Value: unique
  - Distribution of values: Uniform 
  - This is an identifier
  - This value will be used to detect duplicate data point
  - It is a required value
  - Yes. We will use this value to differentiate each trip
  - No, it does not include any sensitive information
- `trip_total`: total amount of money paid for the trip
  - Type: number
  - Default Value: 0
  - Range of Value: 0 - max(trip_total)
  - Distribution of values: Uniform 
  - This is not an identifier
  - This value will be not used to detect duplicate data point
  - It is a required value
  - Yes. We will use this value find the fluctuation of the ride-sharing price
  - No, it does not include any sensitive information
- `trips_pooled`: how many people shared the ride (this is the sum)
  - Type: text
  - Default Value: None
  - Range of Value: unique
  - Distribution of values: Uniform 
  - This is not an identifier
  - This value will not be used to detect duplicate data point
  - It is not a required value
  - Yes. We will use this value to differentiate each trip
  - No, it does not include any sensitive information
  
- `trip_miles`: distance of the trip in miles
  - Type: number
  - Default Value: None
  - Range of Value: 0 - max(trip_miles)
  - Distribution of values: Uniform 
  - This is not an identifier
  - This value will not be used to detect duplicate data point
  - It is a required value
  - Yes. We will use this value to analyze the distance people travel on the ride-sharing service w.r.t the covid-19 spread
  - No, it does not include any sensitive information



## __Data Download__
Original
-  [Chicago COVID-19 Data(csv format)](https://data.cityofchicago.org/api/views/naz8-j4nc/rows.csv?accessType=DOWNLOAD)
-  [Chicago Transportation Network Providers(csv format)](https://data.cityofchicago.org/api/views/m6dm-c72p/rows.csv?accessType=DOWNLOAD)

Ours 
- [Data Available Here](https://drive.google.com/drive/folders/1Dajp9vg09coaEuZZSEOwngf2Xt3W3bDX?usp=sharing)

## __Data sample__ 
-  [Chicago COVID-19 Data](data_deliverable/data/covid/covid_whole.csv)
-  [Chicago Transportation Network Providers](data_deliverable/data/2020/jan.csv)

