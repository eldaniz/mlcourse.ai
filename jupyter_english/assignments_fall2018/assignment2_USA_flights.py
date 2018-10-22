# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 16:58:30 2018

@author: AEGo
"""

#!/usr/bin/env python
# coding: utf-8

# <center>
# <img src="../../img/ods_stickers.jpg" />
#
# ## [mlcourse.ai](mlcourse.ai) – Open Machine Learning Course
#
# Author: [Yury Kashnitskiy](https://yorko.github.io).
# Translated and edited by [Maxim Keremet](https://www.linkedin.com/in/maximkeremet/), [Artem Trunov](https://www.linkedin.com/in/datamove/), and [Aditya Soni](https://www.linkedin.com/in/aditya-soni-0505a9124/). This material is subject to the terms and conditions of the [Creative Commons CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license. Free use is permitted for any non-commercial purpose.

# # <center>Assignment #2. Fall 2018 <br> Exploratory Data Analysis (EDA) of US flights <br> (using Pandas, Matplotlib & Seaborn)
#
# <img src='../../img/plane_sunset.png' width=50%>
#
# Prior to working on the assignment, you'd better check out the corresponding course material:
#  - [Visualization: from Simple Distributions to Dimensionality Reduction](https://mlcourse.ai/notebooks/blob/master/jupyter_english/topic02_visual_data_analysis/topic2_visual_data_analysis.ipynb?flush_cache=true)
#  - [Overview of Seaborn, Matplotlib and Plotly libraries](https://mlcourse.ai/notebooks/blob/master/jupyter_english/topic02_visual_data_analysis/topic2_additional_seaborn_matplotlib_plotly.ipynb?flush_cache=true)
#  - first lectures in [this](https://www.youtube.com/watch?v=QKTuw4PNOsU&list=PLVlY_7IJCMJeRfZ68eVfEcu-UcN9BbwiX) YouTube playlist
#
# ### Your task is to:
#  - write code and perform computations in the cells below
#  - choose answers in the [webform](https://docs.google.com/forms/d/1qSTjLAGqsmpFRhacv0vM-CMQSTT_mtOalNXdRTcdtM0/edit)
#  - submit answers with **the very same email and name** as in assignment 1. This is a part of the assignment, if you don't manage to do so, you won't get credits. If in doubt, you can re-submit A1 form till the deadline for A1, no problem
#
# ### <center> Deadline for A2: 2018 October 21, 20:59 CET
#
#




import numpy as np
import pandas as pd
# pip install seaborn
import seaborn as sns
import matplotlib.pyplot as plt


# * Download the data [archive](http://stat-computing.org/dataexpo/2009/2008.csv.bz2) (Archived ~ 114 Mb, unzipped - ~ 690 Mb). No need to unzip - pandas can unbzip on the fly.
# * Place it in the "../../data" folder, or change the path below according to your location.
# * The dataset has information about carriers and flights between US airports during the year 2008.
# * Column description is available [here](http://www.transtats.bts.gov/Fields.asp?Table_ID=236). Visit this site to find ex. meaning of flight cancellation codes.

# **Reading data into memory and creating a Pandas _DataFrame_ object**
#
# (This may take a while, be patient)
#
# We are not going to read in the whole dataset. In order to reduce memory footprint, we instead load only needed columns and cast them suitable data types.




dtype = {'DayOfWeek': np.uint8, 'DayofMonth': np.uint8, 'Month': np.uint8 , 'Cancelled': np.uint8,
         'Year': np.uint16, 'FlightNum': np.uint16 , 'Distance': np.uint16,
         'UniqueCarrier': str, 'CancellationCode': str, 'Origin': str, 'Dest': str,
         'ArrDelay': np.float16, 'DepDelay': np.float16, 'CarrierDelay': np.float16,
         'WeatherDelay': np.float16, 'NASDelay': np.float16, 'SecurityDelay': np.float16,
         'LateAircraftDelay': np.float16, 'DepTime': np.float16}





get_ipython().run_cell_magic('time', '', "# change the path if needed\npath = '../../data/2008.csv'\nflights_df = pd.read_csv(path, usecols=dtype.keys(), dtype=dtype)")


# **Check the number of rows and columns and print column names.**




print(flights_df.shape)
print(flights_df.columns)


# **Print first 5 rows of the dataset.**




#flights_df.head()
#flights_df[flights_df['CancellationCode'] != 'NaN']['CancellationCode'].head()
flights_df['CancellationCode'] = flights_df['CancellationCode'].astype(str)


# **Transpose the frame to see all features at once.**




flights_df.head().T


# **Examine data types of all features and total dataframe size in memory.**




flights_df.info()


# **Get basic statistics of each feature.**




flights_df.describe().T


# **Count unique Carriers and plot their relative share of flights:**




flights_df['UniqueCarrier'].nunique()





flights_df.groupby('UniqueCarrier').size().plot(kind='bar');


# **We can also _group by_ category/categories in order to calculate different aggregated statistics.**
#
# **For example, finding top-3 flight codes, that have the largest total distance travelled in year 2008.**




flights_df.groupby(['UniqueCarrier','FlightNum'])['Distance'].sum().sort_values(ascending=False).iloc[:3]


# **Another way:**




flights_df.groupby(['UniqueCarrier','FlightNum'])  .agg({'Distance': [np.mean, np.sum, 'count'],
        'Cancelled': np.sum})\
  .sort_values(('Distance', 'sum'), ascending=False)\
  .iloc[0:3]


# **Number of flights by days of week and months:**




pd.crosstab(flights_df.Month, flights_df.DayOfWeek)


# **It can also be handy to color such tables in order to easily notice outliers:**




plt.imshow(pd.crosstab(flights_df.Month, flights_df.DayOfWeek),
           cmap='seismic', interpolation='none');


# **Flight distance histogram:**




flights_df.hist('Distance', bins=20);


# **Making a histogram of flight frequency by date.**




flights_df['Date'] = pd.to_datetime(flights_df.rename(columns={'DayofMonth': 'Day'})[['Year', 'Month', 'Day']])





num_flights_by_date = flights_df.groupby('Date').size()





num_flights_by_date.plot();


# **Do you see a weekly pattern above? And below?**




num_flights_by_date.rolling(window=7).mean().plot();


# **1. Find top-10 carriers in terms of the number of completed flights (_UniqueCarrier_ column)?**
#
# **Which of the listed below is _not_ in your top-10 list?**
# - DL
# - AA
# - OO
# - EV




pp = pd.crosstab(flights_df['UniqueCarrier'], flights_df['Cancelled'])





psorted = pp[1].sort_values(ascending=False)
psorted[:10]


# **2. Plot distributions of flight cancellation reasons (_CancellationCode_).**
#
# **What is the most frequent reason for flight cancellation? (Use this [link](https://www.transtats.bts.gov/Fields.asp?Table_ID=236) to translate codes into reasons)**
# - carrier
# - weather conditions
# - National Air System
# - security reasons




#flights_df['CancellationCode'].unique()
pp = flights_df[flights_df['CancellationCode'] != 'nan']
sns.countplot(x='CancellationCode', data=pp)
cancell_reasons = {'A' : 'Carrier', 'B': 'Weather conditions', 'C': 'National Air System', 'D': 'Security reasons'}
print(cancell_reasons[pp.groupby('CancellationCode').size().sort_values(ascending=False).index[0]])
#!!! print(pp['CancellationCode'].mode())

# **3. Which route is the most frequent, in terms of the number of flights?**
#
# (Take a look at _'Origin'_ and _'Dest'_ features. Consider _A->B_ and _B->A_ directions as _different_ routes)
#
#  - New-York – Washington
#  - San-Francisco – Los-Angeles
#  - San-Jose – Dallas
#  - New-York – San-Francisco




flights_df.groupby(['Origin', 'Dest']).size().idxmax()

#!!! flights_df['Route'] = flights_df['Origin'] + '->' + flights_df['Dest']
#!!! flights_df['Route'].value_counts().head()
#!!! flights_df[['Origin','Dest']].groupby(['Origin','Dest']).size().idxmax()





#flights_df[(flights_df['Origin'] == 'SFO') & (flights_df['Dest'] == 'LAX')].describe()


# **4. Find top-5 delayed routes (count how many times they were delayed on departure). From all flights on these 5 routes, count all flights with weather conditions contributing to a delay.**
#
# - 449
# - 539
# - 549
# - 668



delayColumns = flights_df.columns[flights_df.columns.str.contains('Delay')]

delayed_df = flights_df[flights_df['DepDelay'] > 0]

delayed_df.groupby(['Origin', 'Dest']).size().sort_values(ascending=False)[:5]

top5delayed_df = delayed_df[ (delayed_df['WeatherDelay'] > 0) &
                        (
                                ((delayed_df['Origin'] == 'LAX') & (delayed_df['Dest'] == 'SFO')) |
                                ((delayed_df['Origin'] == 'DAL') & (delayed_df['Dest'] == 'HOU')) |
                                ((delayed_df['Origin'] == 'HOU') & (delayed_df['Dest'] == 'DAL')) |
                                ((delayed_df['Origin'] == 'ORD') & (delayed_df['Dest'] == 'LGA')) |
                                ((delayed_df['Origin'] == 'SFO') & (delayed_df['Dest'] == 'LAX'))
                        )
                        ]
top5delayed_df.shape[0]


# **5. Examine the hourly distribution of departure times. For that, create a
# new series from DepTime, removing missing values.**
#
# **Choose all correct statements:**
#  - Flights are normally distributed within time interval [0-23] (Search for: Normal distribution, bell curve).
#  - Flights are uniformly distributed within time interval [0-23].
#  - In the period from 0 am to 4 am there are considerably less flights than from 7 pm to 8 pm.

depTime_df = flights_df[~np.isnan(flights_df['DepTime'])]
depTime_df['DepTime'] = depTime_df['DepTime'].astype(int)
#!!! depTime_df = (flights_df['DepTime'].dropna() / 100).astype('int')

depTime_df['DepTimeHour'] = (depTime_df['DepTime'] / 100).astype(int)
sns.countplot(x=depTime_df['DepTimeHour'])



# **6. Show how the number of flights changes through time
# (on the daily/weekly/monthly basis) and interpret the findings.**
#
# **Choose all correct statements:**
# - The number of flights during weekends is less than during weekdays (working days).
# - The lowest number of flights is on Sunday.
# - There are less flights during winter than during summer.

sns.countplot(x=flights_df['DayOfWeek'])
sns.countplot(x=flights_df['DayofMonth'])
sns.countplot(x=flights_df['Month'])



# **7. Examine the distribution of cancellation reasons with time.
# Make a bar plot of cancellation reasons aggregated by months.**
#
# **Choose all correct statements:**
# - December has the highest rate of cancellations due to weather.
# - The highest rate of cancellations in September is due to Security reasons.
# - April's top cancellation reason is carriers.
# - Flights cancellations due to National Air System are more frequent than those due to carriers.

cancellation_df = flights_df[flights_df['CancellationCode'] != 'nan']
cancell_reasons = {'A' : 'Carrier', 'B': 'Weather conditions', 'C': 'National Air System', 'D': 'Security reasons'}
cancellation_df['CancellationCode'] = cancellation_df['CancellationCode'].map(cancell_reasons)
cancellation_df = cancellation_df.groupby(['Month', 'CancellationCode'])['Cancelled'].sum()
cancellation_df = cancellation_df.reset_index()

fig, ax = plt.subplots(figsize=(10,10))
sns.barplot(x=cancellation_df['Month'],
            y=cancellation_df['Cancelled'],
            hue=cancellation_df['CancellationCode'],
            data=cancellation_df,
            ax=ax)

cancellation_df.groupby('CancellationCode')['Cancelled'].sum()


# **8. Which month has the greatest number of cancellations due to Carrier?**
# - May
# - January
# - September
# - April ---

#!!!
flights_df.loc[flights_df['CancellationCode'] == 'A', 'Month'].value_counts()


# You code here


# **9. Identify the carrier with the greatest number of cancellations due to
# carrier in the corresponding month from the previous question.**
#
# - 9E
# - EV
# - HA
# - AA


cancellation4_df = flights_df[flights_df['CancellationCode'] == 'A']
cancellation4_df.groupby('UniqueCarrier')['Cancelled'].sum().sort_values(ascending=False)[:1]
#!!! flights_df.loc[(flights_df['CancellationCode'] == 'A') & (flights_df['Month'] == 4),
#!!!               'UniqueCarrier'].value_counts().head()


# **10. Examine median arrival and departure delays (in time) by carrier.
# Which carrier has the lowest median delay time for both arrivals and
# departures? Leave only non-negative values of delay times ('ArrDelay', 'DepDelay').
# [Boxplots](https://seaborn.pydata.org/generated/seaborn.boxplot.html)
# can be helpful in this exercise, as well as it might be a good idea to
# remove outliers in order to build nice graphs. You can exclude delay time
# values higher than a corresponding .95 percentile.**
#
# - EV
# - OO
# - AA
# - AQ

arr_dep_delays_df = flights_df[(flights_df['ArrDelay'] > 0) & (flights_df['DepDelay'] > 0)]
arr_dep_delays_df = arr_dep_delays_df[arr_dep_delays_df.ArrDelay < arr_dep_delays_df.ArrDelay.quantile(.95)]
arr_dep_delays_df = arr_dep_delays_df[arr_dep_delays_df.DepDelay < arr_dep_delays_df.DepDelay.quantile(.95)]
arr_dep_delays_df=arr_dep_delays_df.reset_index()

# box plot
fig, ax = plt.subplots(figsize=(12,10))
sns.boxplot(
        x='UniqueCarrier',
        y='ArrDelay',
        data=arr_dep_delays_df,
        ax=ax
        )
fig, ax = plt.subplots(figsize=(12,10))
sns.boxplot(
        x='UniqueCarrier',
        y='DepDelay',
        data=arr_dep_delays_df,
        ax=ax
        )

arr_dep_delays_df = arr_dep_delays_df.groupby('UniqueCarrier')['ArrDelay', 'DepDelay'].agg(['median'])
arr_dep_delays_df = arr_dep_delays_df.reset_index()
arr_dep_delays_df.columns = ['UniqueCarrier', 'ArrMedian', 'DepMedian']
arr_dep_delays_df.iloc[arr_dep_delays_df['DepMedian'].idxmin()]


#fig, ax = plt.subplots(figsize=(10,10))
#sns.boxplot(
 #       data=arr_dep_delays_df, ax=ax)

