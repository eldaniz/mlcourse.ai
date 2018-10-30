# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 12:07:03 2018

@author: AEGo
"""


# coding: utf-8

# <center>
# <img src="../../img/ods_stickers.jpg" />
#
# ## [mlcourse.ai](mlcourse.ai) – Open Machine Learning Course
# Authors: Yury Isakov, [Yury Kashnitskiy](https://yorko.github.io) (@yorko).
# Edited by Anna Tarelina (@feuerengel). This material is subject to the terms
# and conditions of the
# [Creative Commons CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license. Free use is permitted for any non-commercial purpose.

# # <center> Assignment #4
# ## <center>  User Identification with Logistic Regression (beating
# baselines in the "Alice" competition)
#
# Today we are going to practice working with sparse matrices, training
# Logistic Regression models, and doing feature engineering. We will
# reproduce a couple of baselines in the ["Catch Me If You Can: Intruder
# Detection through Webpage Session Tracking"]
# (https://www.kaggle.com/c/catch-me-if-you-can-intruder-detection-\
# through-webpage-session-tracking2) (a.k.a. "Alice") Kaggle inclass
# competition. More credits will be given for beating a stronger baseline.
#
# **Your task:**
#  1. "Follow me". Complete the missing code and submit your answers via
#    [the google-form](https://docs.google.com/forms/d/1V4lHXkjZvpDDvHAcnH6RuEQJecBaLo8zooxDl1_aP60). 14 credit max. for this part
#  2. "Freeride". Come up with good features to beat the baseline
#   "A4 baseline 3". You need to name your [team](https://www.kaggle.com/c/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2/team) (out of 1 person) in full accordance with the course rating. You can think of it as a part of the assignment. 10 more credits for beating the mentioned baseline and correct team naming.

# # Part 1. Follow me

# <img src='../../img/followme_alice.png' width=50%>
#
# *image credit [@muradosmann](https://www.instagram.com/muradosmann/?hl=en)*

# [1]:


# Import libraries and set desired options
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, GridSearchCV
sns.set()

# -------------------------------------------------------------------------
def get_auc_lr_valid(X, y, C=1.0, seed=17, ratio=0.9):
    # Split the data into the training and validation sets
    idx = int(round(X.shape[0] * ratio))
    # Classifier training
    lr = LogisticRegression(C=C,
                            random_state=seed,
                            solver='liblinear').fit(X[:idx, :], y[:idx])
    # Prediction for validation set
    y_pred = lr.predict_proba(X[idx:, :])[:, 1]
    # Calculate the quality
    score = roc_auc_score(y[idx:], y_pred)

    return score

# -------------------------------------------------------------------------
# Function for writing predictions to a file
def write_to_submission_file(predicted_labels,
                             out_file,
                             target='target',
                             index_label="session_id"):
    predicted_df = pd.DataFrame(predicted_labels,
                                index=np.arange(1,
                                                predicted_labels.shape[0] + 1),
                                columns=[target])
    predicted_df.to_csv(out_file, index_label=index_label)

# -------------------------------------------------------------------------
def do_experiment(data, features, Cs, idx_split):
    # Compose the training set
    experiment = {}
    experiment['features'] = features

    tmp_scaled = StandardScaler().fit_transform(data[experiment['features']])
    X_train = csr_matrix(hstack([full_sites_sparse[:idx_split, :],
                                 tmp_scaled[:idx_split, :]]))

    # Capture the quality with default parameters
    experiment['score_C_default'] = get_auc_lr_valid(X_train, y_train)
    print(experiment['score_C_default'])

    scores = []
    for C in tqdm(Cs):
        scores.append(get_auc_lr_valid(X_train, y_train, C=C))

    experiment['all_Cs'] = Cs
    experiment['all_scores'] = scores
    experiment['score'] = np.array(scores).max()
    experiment['optimal_C'] = Cs[np.array(scores).argmax()]

    plt.plot(Cs, scores, 'ro-')
    plt.xscale('log')
    plt.xlabel('C')
    plt.ylabel('AUC-ROC')
    plt.title('Regularization Parameter Tuning')
    # horizontal line -- model quality with default C value
    plt.axhline(y=experiment['score_C_default'],
                linewidth=.5,
                color='b',
                linestyle='dashed')
    plt.show()

    return (experiment, X_train, y_train)

# ------------------------------------------------------------------------
def make_submission(file_name, X_train, y_train, C, idx_split, random_state=17):
    # Train the model on the whole training data set using optimal
    # regularization parameter
    lr2_2 = LogisticRegression(
            C=C,
            random_state=random_state,
            solver='liblinear').fit(X_train, y_train)

    # Make a prediction for the test set
    X_test = csr_matrix(hstack([full_sites_sparse[idx_split:, :],
                                tmp_scaled[idx_split:, :]]))
    y_test = lr2_2.predict_proba(X_test)[:, 1]

    write_to_submission_file(y_test, file_name)

# import warnings
# warnings.filterwarnings('ignore')

# ##### Problem description
#
# In this competition, we''ll analyze the sequence of websites consequently
# visited by a particular person and try to predict whether this person is
# Alice or someone else. As a metric we will use
# [ROC AUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic).

# ### 1. Data Downloading and Transformation
# Register on [Kaggle](www.kaggle.com), if you have not done it before.
# Go to the competition
# [page](https://inclass.kaggle.com/c/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2) and download the data.
#
# First, read the training and test sets. Then we'll explore the data in
# hand and do a couple of simple exercises.

# [2]:


# Read the training and test data sets, change paths if needed
train_df = pd.read_csv('../../data/websites_train_sessions.csv',
                       index_col='session_id')
test_df = pd.read_csv('../../data/websites_test_sessions.csv',
                      index_col='session_id')

# Convert time1, ..., time10 columns to datetime type
times = ['time%s' % i for i in range(1, 11)]
train_df[times] = train_df[times].apply(pd.to_datetime)
test_df[times] = test_df[times].apply(pd.to_datetime)

# Sort the data by time
train_df = train_df.sort_values(by='time1')

# Look at the first rows of the training set
train_df.head()


# The training data set contains the following features:
#
# - **site1** – id of the first visited website in the session
# - **time1** – visiting time for the first website in the session
# - ...
# - **site10** – id of the tenth visited website in the session
# - **time10** – visiting time for the tenth website in the session
# - **target** – target variable, 1 for Alice's sessions, and 0 for the
#                other users' sessions
#
# User sessions are chosen in the way that they are shorter than 30 min.
# long and contain no more than 10 websites. I.e. a session is considered
# over either if a user has visited 10 websites or if a session has lasted
# over 30 minutes.
#
# There are some empty values in the table, it means that some sessions
# contain less than ten websites. Replace empty values with 0 and change
# columns types to integer. Also load the websites dictionary and check
# how it looks like:

# [3]:


# Change site1, ..., site10 columns type to integer and fill NA-values with
# zeros
sites = ['site%s' % i for i in range(1, 11)]
train_df[sites] = train_df[sites].fillna(0).astype(np.uint16)
test_df[sites] = test_df[sites].fillna(0).astype(np.uint16)

# Load websites dictionary
with open(r"../../data/site_dic.pkl", "rb") as input_file:
    site_dict = pickle.load(input_file)

# Create dataframe for the dictionary
sites_dict = pd.DataFrame(list(site_dict.keys()),
                          index=list(site_dict.values()),
                          columns=['site'])
print(u'Websites total:', sites_dict.shape[0])
sites_dict.head()


# #### 4.1. What are the dimensions of the training and test sets
# (in exactly this order)?
# *For discussions, please stick to
# [ODS Slack](https://opendatascience.slack.com/), channel #mlcourse_ai,
# pinned thread __#a4_q1__*
#
# - (82797, 20) and (253561, 20)
# - (82797, 20) and (253561, 21)
# - (253561, 21) and (82797, 20)
# - (253561, 20) and (82797, 20) ---

# [4]:


print(train_df.shape, test_df.shape)


# ### 2. Brief Exploratory Data Analysis

# Before we start training models, we have to perform Exploratory Data
#  Analysis ([EDA](https://en.wikipedia.org/wiki/Exploratory_data_analysis)).
# Today, we are going to perform a shorter version, but we will use other
# techniques as we move forward. Let's check which websites in the training
# data set are the most visited. As you can see, they are Google services
# and a bioinformatics website (a website with 'zero'-index is our missed
# values, just ignore it):

# [19]:


# Top websites in the training data set
top_sites = pd.Series(train_df[sites].values.flatten()).value_counts().\
    sort_values(ascending=False).head(5)
print(top_sites)
sites_dict.loc[top_sites.drop(0).index]

# ##### 4.2. What kind of websites does Alice visit the most?
# *For discussions, please stick to
#  [ODS Slack](https://opendatascience.slack.com/), channel #mlcourse_ai,
# pinned thread __#a4_q2__*
#
# - videohostings  -- ??? i1.ytimg.com
# - social networks
# - torrent trackers
# - news



top_sites_alice = pd.Series(
        train_df[train_df['target'] == 1][sites].values.flatten()).\
        value_counts().sort_values(ascending=False).head(5)
print(top_sites_alice)
print(sites_dict.loc[top_sites_alice.index])
print(sites_dict.loc[top_sites_alice.index[0]])  # ??? i1.ytimg.com

print('\nThe most visited site: {0}'.
      format(sites_dict.loc[top_sites_alice.index[0]].site))  # ?? i1.ytimg.com


# Now let us look at the timestamps and try to characterize sessions
# as timeframes:

# [21]:


# Create a separate dataframe where we will work with timestamps
time_df = pd.DataFrame(index=train_df.index)
time_df['target'] = train_df['target']

# Find sessions' starting and ending
time_df['min'] = train_df[times].min(axis=1)
time_df['max'] = train_df[times].max(axis=1)

# Calculate sessions' duration in seconds
time_df['seconds'] = (time_df['max'] - time_df['min']) / np.timedelta64(1, 's')

time_df.head()

# In order to perform the next task, generate descriptive statistics as you
# did in the first assignment.
#
# ##### 4.3. Select all correct statements:
# *For discussions, please stick to
# [ODS Slack](https://opendatascience.slack.com/), channel #mlcourse_ai,
# pinned thread __#a4_q3__*
#
# - on average, Alice's session is shorter than that of other users ---yes
# - more than 1% of all sessions in the dataset belong to Alice  ---no
# - minimum and maximum durations of Alice's and other users' sessions are approximately the same ---no???
# - variation about the mean session duration for all users (including Alice) is approximately the same ---no
# - less than a quarter of Alice's sessions are greater than or equal to 40 seconds --yes

# [22]:

print(time_df.groupby('target')['seconds'].describe())

# 1.
print(time_df.groupby('target')['seconds'].mean())  # 0: 139.3;  1: 52.3

# 2.
print(time_df['target'].value_counts(normalize=True))  # 0.009059

# 3.
print(time_df.groupby('target')['seconds'].agg([min, max]))  # max_alice=1763   max_intruder=1800

# 4.
print(time_df.groupby('target')['seconds'].mean())  # 0: 139.3;  1: 52.3

# 5.
print(time_df[(time_df['target'] == 1) & (time_df['seconds'] >= 40)].shape[0] *
              100 / time_df[time_df['target'] == 1].shape[0])  # 24.11 %



# In order to train our first model, we need to prepare the data. First of
# all, exclude the target variable from the training set. Now both training
# and test sets have the same number of columns, therefore aggregate them
# into one dataframe.  Thus, all transformations will be performed
# simultaneously on both training and test data sets.
#
# On the one hand, it leads to the fact that both data sets have one feature
# space (you don't have to worry that you forgot to transform a feature in
# some data sets). On the other hand, processing time will increase.
# For the enormously large sets it might turn out that it is impossible to
# transform both data sets simultaneously (and sometimes you have to split
# your transformations into several stages only for train/test data set).
# In our case, with this particular data set, we are going to perform all
# the transformations for the whole united dataframe at once, and before
# training the model or making predictions we will just take its appropriate
# part.

# [23]:


# Our target variable
y_train = train_df['target']

# United dataframe of the initial data
full_df = pd.concat([train_df.drop('target', axis=1), test_df])

# Index to split the training and test data sets
idx_split = train_df.shape[0]


# For the very basic model, we will use only the visited websites in the
# session (but we will not take into account timestamp features). The point
# behind this data selection is: *Alice has her favorite sites, and the more
# often you see these sites in the session, the higher probability that this
# is Alice's session, and vice versa.*
#
# Let us prepare the data, we will take only features
# `site1, site2, ... , site10` from the whole dataframe. Keep in mind that
# the missing values are replaced with zero. Here is how the first rows of
# the dataframe look like:

# [24]:


# Dataframe with indices of visited websites in session
full_sites = full_df[sites]
full_sites.head()


# Sessions are sequences of website indices, and data in this representation
# is useless for machine learning method (just think, what happens if we
# switched all ids of all websites).
#
# According to our hypothesis (Alice has favorite websites), we need to
# transform this dataframe so each website has a corresponding feature
# (column) and its value is equal to number of this website visits in the
# session. It can be done in two lines:

# [25]:


# sequence of indices
sites_flatten = full_sites.values.flatten()

# and the matrix we are looking for
# (make sure you understand which of the `csr_matrix` constructors is used here)
# a further toy example will help you with it
full_sites_sparse = csr_matrix(([1] * sites_flatten.shape[0],
                                sites_flatten,
                                range(0,
                                      sites_flatten.shape[0] + 10, 10)))[:, 1:]


# [26]:


full_sites_sparse.shape


# If you understand what just happened here, then you can skip the next
# passage (perhaps, you can handle logistic regression too?), If not, then
# let us figure it out.
#
# ### Important detour #1: Sparse Matrices
#
# Let us estimate how much memory it will require to store our data in the
# example above. Our united dataframe contains 336 thousand samples of 48
# thousand integer features in each. It's easy to calculate the required
# amount of memory, roughly:
#
# $$336K * 48K * 8 bytes = 16M * 8 bytes = 128 GB,$$
#
# (that's the
# [exact](http://www.wolframalpha.com/input/?i=336358*48371*8+bytes) value).
# Obviously, ordinary mortals have no such volumes (strictly speaking,
# Python may allow you to create such a matrix, but it will not be easy
# to do anything with it). The interesting fact is that most of the elements
# of our matrix are zeros. If we count non-zero elements, then it will be
# about 1.8 million, i.е. slightly more than 10% of all matrix elements.
# Such a matrix, where most elements are zeros, is called sparse, and the
# ratio between the number of zero elements and the total number of elements
# is called the sparseness of the matrix.
#
# For the work with such matrices you can use `scipy.sparse` library, check
# [documentation]
# (https://docs.scipy.org/doc/scipy-0.18.1/reference/sparse.html) to
# understand what possible types of sparse matrices are, how to work with
# them and in which cases their usage is most effective. You can learn how
# they are arranged, for example, in Wikipedia
# [article](https://en.wikipedia.org/wiki/Sparse_matrix).
# Note, that a sparse matrix contains only non-zero elements, and you can
# get the allocated memory size like this (significant memory savings
# are obvious):

# [27]:


# How much memory does a sparse matrix occupy?
print('{0} elements * {1} bytes = {2} bytes'.
      format(full_sites_sparse.count_nonzero(),
             8,
             full_sites_sparse.count_nonzero() * 8))

# Or just like this:
print('sparse_matrix_size = {0} bytes'.format(full_sites_sparse.data.nbytes))


# Let us explore how the matrix with the websites has been formed using a
# mini example. Suppose we have the following table with user sessions:
#
# | id | site1 | site2 | site3 |
# |---|---|---|---|
# | 1 | 1 | 0 | 0 |
# | 2 | 1 | 3 | 1 |
# | 3 | 2 | 3 | 4 |
#
# There are 3 sessions, and no more than 3 websites in each. Users visited
# four different sites in total (there are numbers from 1 to 4 in the table
# cells). And let us assume that the mapping is:
#
#  1. vk.com
#  2. habrahabr.ru
#  3. yandex.ru
#  4. ods.ai
#
# If the user has visited less than 3 websites during the session, the last
# few values will be zero. We want to convert the original dataframe in a way
# that each session has a corresponding row which shows the number of visits
# to each particular site. I.e. we want to transform the previous table into
# the following form:
#
# | id | vk.com | habrahabr.ru | yandex.ru | ods.ai |
# |---|---|---|---|---|
# | 1 | 1 | 0 | 0 | 0 |
# | 2 | 2 | 0 | 1 | 0 |
# | 3 | 0 | 1 | 1 | 1 |
#
#
# To do this, use the constructor: `csr_matrix ((data, indices, indptr))`
# and create a frequency table (see examples, code and comments on the links
# above to see how it works). Here we set all the parameters explicitly for
# greater clarity:

# [28]:


# data, create the list of ones, length of which equal to the number of
# elements in the initial dataframe (9)
# By summing the number of ones in the cell, we get the frequency,
# number of visits to a particular site per session
data = [1] * 9

# To do this, you need to correctly distribute the ones in cells
# Indices - website ids, i.e. columns of a new matrix. We will sum ones up
# grouping them by sessions (ids)
indices = [1, 0, 0, 1, 3, 1, 2, 3, 4]

# Indices for the division into rows (sessions)
# For example, line 0 is the elements between the indices [0; 3) - the
# rightmost value is not included
# Line 1 is the elements between the indices [3; 6)
# Line 2 is the elements between the indices [6; 9)
indptr = [0, 3, 6, 9]

# Aggregate these three variables into a tuple and compose a matrix
# To display this matrix on the screen transform it into the usual "dense"
# matrix
csr_matrix((data, indices, indptr)).todense()


# As you might have noticed, there are not four columns in the resulting
# matrix (corresponding to number of different websites) but five. A zero
# column has been added, which indicates if the session was shorter
# (in our mini example we took sessions of three). This column is excessive
# and should be removed from the dataframe (do that yourself).
#
# ##### 4.4. What is the sparseness of the matrix in our small example?
# *For discussions, please stick to
# [ODS Slack](https://opendatascience.slack.com/), channel #mlcourse_ai,
# pinned thread __#a4_q4__*
#
# - 42%
# - 47%
# - 50% ---
# - 53%
#
#
#

# [29]:


sp_arr = csr_matrix((data, indices, indptr)).todense()
print(sp_arr)
sp_arr = np.delete(sp_arr, 0, axis=1)
print(sp_arr)
print(sp_arr.nonzero()[0].shape[0])
sp_arr.nonzero()[0].shape[0] / (sp_arr.shape[0] * sp_arr.shape[1]) * 100  # 50


# Another benefit of using sparse matrices is that there are special
# implementations of both matrix operations and machine learning algorithms
# for them, which sometimes allows to significantly accelerate operations
# due to the data structure peculiarities. This applies to logistic regression
# as well. Now everything is ready to build our first model.
#
# ### 3. Training the first model
#
# So, we have an algorithm and data for it. Let us build our first model,
# using [logistic regression]
# (http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
# implementation from ` Sklearn` with default parameters. We will use the
# first 90% of the data for training (the training data set is sorted
# by time), and the remaining 10% for validation. Let's write a simple
# function that returns the quality of the model and then train our first
# classifier:

# [30]:

# [31]:


# Select the training set from the united dataframe (where we have the answers)
X_train = full_sites_sparse[:idx_split, :]

# Calculate metric on the validation set
print(get_auc_lr_valid(X_train, y_train))


# The first model demonstrated the quality  of 0.92 on the validation set.
# Let's take it as the first baseline and starting point. To make a
# prediction on the test data set **we need to train the model again on
# the entire training data set** (until this moment, our model used only
# part of the data for training), which will increase its generalizing ability:

# [32]:


# Train the model on the whole training data set
# Use random_state=17 for repeatability
# Parameter C=1 by default, but here we set it explicitly
lr = LogisticRegression(C=1.0,
                        random_state=17,
                        solver='liblinear').fit(X_train, y_train)

# Make a prediction for test data set
X_test = full_sites_sparse[idx_split:, :]
y_test = lr.predict_proba(X_test)[:, 1]

# Write it to the file which could be submitted
write_to_submission_file(y_test, 'baseline_1.csv')


# If you follow these steps and upload the answer to the competition
# [page](https://inclass.kaggle.com/c/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2),
# you will get `ROC AUC = 0.90812` on the public leaderboard ("A4 baseline 1").
#
# ### 4. Model Improvement: Feature Engineering
#
# Now we are going to try to improve the quality of our model by adding new
# features to the data. But first, answer the following question:
#
# ##### 4.5. What years are present in the training and test datasets,
# if united?
# *For discussions, please stick to
# [ODS Slack](https://opendatascience.slack.com/), channel #mlcourse_ai,
# pinned thread __#a4_q5__*
#
# - 13 and 14
# - 2012 and 2013
# - 2013 and 2014  ---
# - 2014 and 2015

# [34]:

def min_max(ts):
    return ts.apply(lambda ts: ts.year).astype('float64').agg(['min', 'max'])


dd = pd.DataFrame(full_df[times].apply(min_max).astype('float64'))
print('min: {}'.format(np.min(dd.min(axis=0))))  # 2013
print('max: {}'.format(np.max(dd.max(axis=0))))  # 2014


# Create a feature that will be a number in YYYYMM format from the date
# when the session was held, for example 201407 -- year 2014 and 7th month.
# Thus, we will take into account the monthly
# [linear trend](http://people.duke.edu/~rnau/411trend.htm) for the entire
# period of the data provided.

# [35]:


# Dataframe for new features
full_new_feat = pd.DataFrame(index=full_df.index)

# Add start_month feature
full_new_feat['start_month'] = \
    full_df['time1'].\
    apply(lambda ts: 100 * ts.year + ts.month).astype('float64')


# ##### 4.6. Plot the graph of the number of Alice sessions versus the new
# feature, start_month. Choose the correct statement:
#
# *For discussions, please stick to
# [ODS Slack](https://opendatascience.slack.com/), channel #mlcourse_ai,
# pinned thread __#a4_q6__*
#
# - Alice wasn't online at all for the entire period
# - From the beginning of 2013 to mid-2014, the number of Alice's sessions per month decreased
# - The number of Alice's sessions per month is generally constant for the entire period
# - From the beginning of 2013 to mid-2014, the number of Alice's sessions per month increased ---???
#
# *Hint: the graph will be more explicit if you treat `start_month` as a
# categorical ordinal variable*.

# [36]:

fig, ax = plt.subplots(figsize=(14, 6))
plt.xticks(rotation=70)
sns.countplot(
        x=full_new_feat['start_month'],
        ax=ax)
plt.ylabel('Sessions count')
plt.title('All sessions');


train_new_feat = pd.DataFrame(index=train_df.index)

train_new_feat['target'] = train_df['target']

# Add start_month feature
train_new_feat['start_month'] = train_df['time1'].\
    apply(lambda ts: 100 * ts.year + ts.month).astype('float64')

fig, ax = plt.subplots(figsize=(10, 6))
# plt.xticks(rotation=70)
sns.countplot(
        x=train_new_feat[train_new_feat['target'] == 1]['start_month'],
        ax=ax)
plt.ylabel('Sessions count')
plt.title('Alice sessions count');

# all intruders train session
fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(x=train_new_feat[train_new_feat['target'] == 0]['start_month'],
              ax=ax)
plt.ylabel('Intruder sessions count')
plt.xticks(rotation=70);

# In this way, we have an illustration and thoughts about the usefulness
# of the new feature, add it to the training sample and check the quality
# of the new model:

# [37]:


# Add the new feature to the sparse matrix
tmp = full_new_feat[['start_month']].values
X_train = csr_matrix(hstack([full_sites_sparse[:idx_split, :],
                             tmp[:idx_split, :]]))

# Compute the metric on the validation set
print(get_auc_lr_valid(X_train, y_train))


# The quality of the model has decreased significantly. We added a feature
# that definitely seemed useful to us, but its usage only worsened the model.
# Why did it happen?
#
# ### Important detour #2: is it necessary to scale features?
#
# Here we give an intuitive reasoning (a rigorous mathematical justification
# for one or another aspect in linear models you can easily find on the
# internet). Consider the features more closely: those of them that
# correspond to the number of visits to a particular web-site per session
# vary from 0 to 10. The feature `start_month` has a completely different
# range: from 201301 to 201412, this means the contribution of this variable
# is significantly greater than the others. It would seem that problem can be
# avoided if we put less weight in a linear combination of attributes in this
# case, but in our case logistic regression with regularization is used
# (by default, this parameter is `C = 1`), which penalizes the model the
# stronger the greater its weights are. Therefore, for linear methods with
# regularization, it is recommended to convert features to the same scale
# (you can read more about the regularization, for example,
# [here](https://habrahabr.ru/company/ods/blog/322076/)).
#
# One way to do this is standardization: for each observation you need to
# subtract the average value of the feature and divide this difference by
# the standard deviation:
#
# $$ x^{*}_{i} = \dfrac{x_{i} - \mu_x}{\sigma_x}$$
#
# The following practical tips can be given:
# - It is recommended to scale features if they have essentially different
#   ranges or different units of measurement (for example, the country's
#   population is indicated in units, and the country's GNP in trillions)
# - Scale features if you do not have a reason/expert opinion to give a
#   greater weight to any of them
# - Scaling can be excessive if the ranges of some of your features differ
#   from each other, but they are in the same system of units (for example,
#   the proportion of middle-aged people and people over 80 among the entire
#   population)
# - If you want to get an interpreted model, then build a model without
#   regularization and scaling (most likely, its quality will be worse)
# - Binary features (which take only values of 0 or 1) are usually left
#   without conversion, (but)
# - If the quality of the model is crucial, try different options and select
#   one where the quality is better
#
# Getting back to `start_month`, let us rescale the new feature and train
# the model again. This time the quality has increased:

# [38]:


# Add the new standardized feature to the sparse matrix
tmp = StandardScaler().fit_transform(full_new_feat[['start_month']])
X_train = csr_matrix(hstack([full_sites_sparse[:idx_split, :],
                             tmp[:idx_split, :]]))

# Compute metric on the validation set
acc_scaler_start_month = get_auc_lr_valid(X_train, y_train)
print(acc_scaler_start_month)


# ##### 4.7. Add to the training set a new feature "n_unique_sites" – the
# number of the unique web-sites in a session. Calculate how the quality on
# the validation set has changed
#
# *For discussions, please stick to
# [ODS Slack](https://opendatascience.slack.com/), channel #mlcourse_ai,
# pinned thread __#a4_q7__*
#
# - It has decreased. It is better not to add a new feature. ---???
# - It has not changed
# - It has decreased. The new feature should be scaled.
# - I am confused, and I do not know if it's necessary to scale a new feature.
#
# *Tips: use the nunique() function from `pandas`. Do not forget to include
# the start_month in the set. Will you scale a new feature? Why?*

# [39]:


# Add start_month feature
full_new_feat['n_unique_sites'] = \
    full_df[full_df[sites] != 0][sites].apply(
            lambda site: site[site != 0].nunique(),
            axis=1).astype('float64')

tmp_months = StandardScaler().fit_transform(full_new_feat[['start_month']])
tmp_uniqes = full_new_feat[['n_unique_sites']].values
X_train = csr_matrix(hstack([full_sites_sparse[:idx_split, :],
                             tmp_months[:idx_split, :],
                             tmp_uniqes[:idx_split, :]]))

# Compute the metric on the validation set
acc_unique_sites = get_auc_lr_valid(X_train, y_train)
print('Quality with unique sites feature {}'.format(acc_unique_sites))
print('Is quality with unique sites greater? : {}'.
      format(acc_unique_sites > acc_scaler_start_month))


# with scaler unique
tmp_uniqes = StandardScaler().fit_transform(full_new_feat[['n_unique_sites']])
X_train = csr_matrix(hstack([full_sites_sparse[:idx_split, :],
                             tmp_months[:idx_split, :],
                             tmp_uniqes[:idx_split, :]]))

# Compute the metric on the validation set
acc_scaler_unique_sites = get_auc_lr_valid(X_train, y_train)
print('Quality with SCALED unique sites feature {}'.format(acc_scaler_unique_sites))
print('Is quality greater? : {}'.
      format(acc_scaler_unique_sites > acc_unique_sites))



# So, the new feature has slightly decreased the quality, so we will not use
# it. Nevertheless, do not rush to throw features out because they haven't
# performed well. They can be useful in a combination with other features
# (for example, when a new feature is a ratio or a product of two others).
#
# #####  4.8. Add two new features: start_hour and morning. Calculate the
# metric. Which of these features gives an improvement?
#
# The `start_hour` feature is the hour at which the session started
# (from 0 to 23), and the binary feature `morning` is equal to 1 if the
# session started in the morning and 0 if the session started later
# (we assume that morning means `start_hour` is equal to 11 or less).
#
# Will you scale the new features? Make your assumptions and test them
# in practice.
#
# *For discussions, please stick to
# [ODS Slack](https://opendatascience.slack.com/), channel #mlcourse_ai,
# pinned thread __#a4_q8__*
#
# - None of the features gave an improvement :(
# - `start_hour` feature gave an improvement, and `morning` did not
# - `morning` feature gave an improvement, and `start_hour` did not
# - Both features gave an improvement ---
#
# *Tip: find suitable functions for working with time series data in
# [documentation](http://pandas.pydata.org/pandas-docs/stable/api.html).
# Do not forget to include the `start_month` feature.*

# [40]:


full_new_feat['start_hour'] = \
    full_df['time1'].apply(lambda ts: ts.hour).astype('float64')
full_new_feat['morning'] = (full_new_feat['start_hour'] <= 11).astype('int')

tmp_start_month_scaled = StandardScaler().fit_transform(full_new_feat[['start_month']])

scores = {}
# 'start_hour' feature
tmp_start_hour = full_new_feat[['start_hour']].values
X_train = csr_matrix(hstack([full_sites_sparse[:idx_split, :],
                             tmp_start_month_scaled[:idx_split, :],
                             tmp_start_hour[:idx_split, :]]))
scores['score_start_hour_only'] = get_auc_lr_valid(X_train, y_train)
print(scores['score_start_hour_only'])

# 'start_hour' feature SCALED
tmp_start_hour_scaled = StandardScaler().fit_transform(full_new_feat[['start_hour']])
X_train = csr_matrix(hstack([full_sites_sparse[:idx_split, :],
                             tmp_start_month_scaled[:idx_split, :],
                             tmp_start_hour_scaled[:idx_split, :]]))
scores['score_start_hour_only_scaled'] = get_auc_lr_valid(X_train, y_train)
print(scores['score_start_hour_only_scaled'])

# 'morning' feature
tmp_morning = full_new_feat[['morning']].values
X_train = csr_matrix(hstack([full_sites_sparse[:idx_split, :],
                             tmp_start_month_scaled[:idx_split, :],
                             tmp_morning[:idx_split, :]]))
scores['score_morning_only'] = get_auc_lr_valid(X_train, y_train)
print(scores['score_morning_only'])

# 'morning' feature SCALED
tmp_morning_scaled = StandardScaler().fit_transform(full_new_feat[['morning']]);
X_train = csr_matrix(hstack([full_sites_sparse[:idx_split, :],
                             tmp_start_month_scaled[:idx_split, :],
                             tmp_morning_scaled[:idx_split, :]]));
scores['score_morning_only_scaled'] = get_auc_lr_valid(X_train, y_train);
print(scores['score_morning_only_scaled']);

# both
X_train = csr_matrix(hstack([full_sites_sparse[:idx_split, :],
                             tmp_start_month_scaled[:idx_split, :],
                             tmp_start_hour[:idx_split, :],
                             tmp_morning[:idx_split, :]]))
scores['score_both_morning_and_starthour'] = get_auc_lr_valid(X_train, y_train)
print(scores['score_both_morning_and_starthour'])

# both scaled
X_train = csr_matrix(hstack([full_sites_sparse[:idx_split, :],
                             tmp_start_month_scaled[:idx_split, :],
                             tmp_start_hour_scaled[:idx_split, :],
                             tmp_morning_scaled[:idx_split, :]]))
scores['score_both_morning_and_starthour_scaled'] = get_auc_lr_valid(X_train, y_train)
print(scores['score_both_morning_and_starthour_scaled'])

scores_df = pd.DataFrame(data = list(scores.values()), index = scores.keys(), columns=['score'])
scores_df['score'].sort_values(ascending=False)



# ### 5. Regularization and Parameter Tuning
#
# We have introduced features that improve the quality of our model in comparison with the first baseline. Can we do even better? After we have changed the training and test sets, it almost always makes sense to search for the optimal hyperparameters - the parameters of the model that do not change during training.
#
# For example, in week 3, you learned that, in decision trees, the depth of the tree is a hyperparameter, but the feature by which splitting occurs and its threshold is not.
#
# In the logistic regression that we use, the weights of each feature are changing, and we find their optimal values during training; meanwhile, the regularization parameter remains constant. This is the hyperparameter that we are going to optimize now.
#
# Calculate the quality on a validation set with a regularization parameter, which is equal to 1 by default:

# [ ]:


# Compose the training set
tmp_scaled = StandardScaler().fit_transform(full_new_feat[['start_month',
                                                           'start_hour',
                                                           'morning']])
X_train = csr_matrix(hstack([full_sites_sparse[:idx_split,:],
                             tmp_scaled[:idx_split,:]]))

# Capture the quality with default parameters
score_C_1 = get_auc_lr_valid(X_train, y_train)
print(score_C_1)

# We will try to beat this result by optimizing the regularization parameter. We will take a list of possible values of C and calculate the quality metric on the validation set for each of C-values:

# [ ]:


from tqdm import tqdm

# List of possible C-values
Cs = np.logspace(-3, 1, 10)
scores = []
for C in tqdm(Cs):
    scores.append(get_auc_lr_valid(X_train, y_train, C=C))


# Plot the graph of the quality metric (AUC-ROC) versus the value of the regularization parameter. The value of quality metric corresponding to the default value of C=1 is represented by a horizontal dotted line:

# [ ]:


plt.plot(Cs, scores, 'ro-')
plt.xscale('log')
plt.xlabel('C')
plt.ylabel('AUC-ROC')
plt.title('Regularization Parameter Tuning')
# horizontal line -- model quality with default C value
plt.axhline(y=score_C_1, linewidth=.5, color='b', linestyle='dashed')
plt.show()

# ##### 4.9. What is the value of parameter C (if rounded to 2 decimals) that corresponds to the highest model quality?
#
# *For discussions, please stick to [ODS Slack](https://opendatascience.slack.com/), channel #mlcourse_ai, pinned thread __#a4_q9__*
#
# - 0.17  ---
# - 0.46
# - 1.29
# - 3.14

# [ ]:

optimal_c = Cs[np.array(scores).argmax()]
print(round(float(optimal_c), 2))
score_C_1_optimal = np.array(scores).max()
# For the last task in this assignment: train the model using the optimal regularization parameter you found (do not round up to two digits like in the last question). If you do everything correctly and submit your solution, you should see `ROC AUC = 0.92784` on the public leaderboard ("A4 baseline 2"):

# [ ]:


# Prepare the training and test data
tmp_scaled = StandardScaler().fit_transform(full_new_feat[['start_month',
                           'start_hour',
                           'morning']])
X_train = csr_matrix(hstack([full_sites_sparse[:idx_split, :],
                             tmp_scaled[:idx_split,:]]))
X_test = csr_matrix(hstack([full_sites_sparse[idx_split:, :],
                            tmp_scaled[idx_split:,:]]))

# Train the model on the whole training data set using optimal regularization parameter
lr = LogisticRegression(
        C=optimal_c,
        random_state=17,
        solver='liblinear').fit(X_train, y_train)

# Make a prediction for the test set
y_test = lr.predict_proba(X_test)[:, 1]

# Write it to the submission file
write_to_submission_file(y_test, 'baseline_2.csv')
optimal_c1 = optimal_c


# -------------------------------------------------------------------------
#
# -------------------------------------------------------------------------
experiments = {}

# -------------------------------------------------------------------------
experiment_name = 'baseline_2'
experiment, X_train, y_train = do_experiment(data=full_new_feat,
                                             features=['start_month',
                                                       'start_hour',
                                                       'morning'],
                                             Cs=np.logspace(-3, 1, 10),
                                             idx_split=idx_split)

print('best score: {}\ndefault score:{}'.
      format(
              experiment['score'],
              experiment['score_C_default']))
round(float(experiment['optimal_C']), 2)

experiment['submission_file'] = experiment_name + '.csv'
experiments[experiment_name] = experiment

make_submission(experiment['submission_file'],
                X_train,
                y_train,
                experiments[experiment_name]['optimal_C'],
                idx_split=idx_split
                )

# -------------------------------------------------------------------------

# In this part of the assignment, you have learned how to use sparse matrices, train logistic regression models, create new features and selected the best ones, learned why you need to scale features, and how to select hyperparameters. That's a lot!

# # Part 2. Freeride

# <img src='../../img/snowboard.jpg' width=70%>
#
# *Yorko in Sheregesh, the best palce in Russia for snowboarding and skiing.*

# In this part, you'll need to beat the "A4 baseline 3" baseline. No more step-by-step instructions. But it'll be very helpful for you to study the Kernel "[Correct time-aware cross-validation scheme](https://www.kaggle.com/kashnitsky/correct-time-aware-cross-validation-scheme)".
#
# Here are a few tips for finding new features: think about what you can come up with using existing features, try multiplying or dividing two of them, justify or decline your hypotheses with plots, extract useful information from time series data (time1 ... time10), do not hesitate to convert an existing feature (for example, take a logarithm), etc. Checkout other [Kernels](https://www.kaggle.com/c/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2/kernels). We encourage you to try new ideas and models throughout the course and participate in the competitions - it's fun!
#
# When you get into Kaggle and Xgboost, you'll feel like that, and it's OK :)
#
# <img src='../../img/xgboost_meme.jpg' width=50%>


# -------------------------------------------------------------------------
experiment_name = 'baseline_2_2'

#full_new_feat['morning'] = (7 <= full_new_feat['start_hour'] <= 11).astype('int')
full_new_feat['day'] = \
    ((12 <= full_new_feat['start_hour']) &
     (full_new_feat['start_hour'] <= 18)).astype('int')
full_new_feat['evening'] = \
    ((19 <= full_new_feat['start_hour']) &
     (full_new_feat['start_hour'] <= 23)).astype('int')

experiment, X_train, y_train = do_experiment(data=full_new_feat,
                                             features=['start_month',
                                                       'start_hour',
                                                       'evening',
                                                       'morning'],
                                             Cs=np.logspace(-3, 1, 10),
                                             idx_split=idx_split)

print('best score: {}\ndefault score:{}'.
      format(
              experiment['score'],
              experiment['score_C_default']))
round(float(experiment['optimal_C']), 2)

experiment['submission_file'] = experiment_name + '.csv'
experiments[experiment_name] = experiment

make_submission(experiment['submission_file'],
                X_train,
                y_train,
                experiments[experiment_name]['optimal_C'],
                idx_split=idx_split
                )

# -------------------------------------------------------------------------


logit_params = {
        'solver': ['liblinear', 'sag', 'newton-cg'],
        'C': Cs}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)

logit_grid = GridSearchCV(
        LogisticRegression(random_state=17),
        logit_params,
        scoring='roc_auc',
        cv=skf,
        n_jobs=-1,
        verbose=True)

logit_grid.fit(X_train, y_train)
print(logit_grid.best_estimator_)
print(logit_grid.best_score_)

# Make a prediction for the test set
y_test = logit_grid.predict_proba(X_test)[:, 1]

write_to_submission_file(y_test, 'baseline_2_3.csv')





tmp_scaled = StandardScaler().fit_transform(full_new_feat[['start_month',
                                                           'start_hour',
                                                           'morning']])
X_train = csr_matrix(hstack([full_sites_sparse[:idx_split, :],
                             tmp_scaled[:idx_split,:]]))
X_test = csr_matrix(hstack([full_sites_sparse[idx_split:, :],
                            tmp_scaled[idx_split:, :]]))



logit_params = {
        'solver': ['liblinear', 'sag', 'newton-cg'],
        'C': Cs}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)

logit_grid = GridSearchCV(
        LogisticRegression(random_state=17),
        logit_params,
        scoring='roc_auc',
        cv=skf,
        n_jobs=-1,
        verbose=True)

logit_grid.fit(X_train, y_train)
logit_grid.best_estimator_
logit_grid.best_score_

# Make a prediction for the test set
y_test = logit_grid.predict_proba(X_test)[:, 1]

write_to_submission_file(y_test, 'baseline_2_4_3features_newton.csv')



import xgboost as xgb

#dtrain_X = xgb.DMatrix(X_train, label=y_train)
#
#dtest_X = xgb.DMatrix(X_test)
#
#param = {
#        'max_depth': 2,
#        'eval_metric': 'auc',
#        'eta': 1,
#        'silent': 1,
#        'objective': 'binary:logistic'}
#
#num_round = 10
##evallist = [(dtest_X, 'eval'), (dtrain_X, 'train')]
#evallist = [(dtrain_X, 'train')]
##bst = xgb.train(param, dtrain_X, num_round, evallist)
#bst = xgb.cv(param, dtrain_X, num_round, nfold=5, seed=17)
#xgb.plot_importance(bst)
##bst = xgb.train(param, dtrain_X, num_round, evallist)



from xgboost.sklearn import XGBClassifier

xgb_params = {
              'objective':['binary:logistic'],
              'learning_rate': [0.05, 0.1, 0.02], #so called `eta` value
              'max_depth': [2, 5, 6, 10, 20],
              'min_child_weight': [11],
              'silent': [1],
              'subsample': [0.8],
              'colsample_bytree': [0.7],
              'n_estimators': [10, 20, 50, 200], #number of trees, change it to 1000 for better results
              'random_state': [17]}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)

xgb_grid = GridSearchCV(
        XGBClassifier(random_state=17),
        xgb_params,
        scoring='roc_auc',
        cv=skf,
        n_jobs=-1,
        verbose=True,
        return_train_score=True)

xgb_grid.fit(X_train, y_train)

print(xgb_grid.best_estimator_)
print(xgb_grid.best_params_)
print('[Test] Max ROC_AUC value: {}'.format(xgb_grid.best_score_))
#print('Best std score: {}'.
#      format(xgb_grid.cv_results_['std_test_score'][xgb_grid.best_index_]))
print('[Train] Max ROC_AUC value: {}'.
      format(xgb_grid.cv_results_['mean_train_score'][xgb_grid.best_index_]))
print('Max ROC_AUC value: {}'.format(xgb_grid.best_score_))


# Make a prediction for the test set
y_test = xgb_grid.predict_proba(X_test)[:, 1]

write_to_submission_file(y_test, 'baseline_2_5_xgb.csv')






xgb_params = {
              'objective':['binary:logistic'],
              'learning_rate': [0.05, 0.1, 0.02], #so called `eta` value
              'max_depth': [2, 5, 6, 10, 20],
              'min_child_weight': [11],
              'silent': [1],
              'subsample': [0.8],
              'colsample_bytree': [0.7],
              'n_estimators': [10, 20, 50, 200], #number of trees, change it to 1000 for better results
              'random_state': [17]}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)

xgb_grid = GridSearchCV(
        XGBClassifier(random_state=17),
        xgb_params,
        scoring='roc_auc',
        cv=skf,
        n_jobs=-1,
        verbose=True,
        return_train_score=True)

xgb_grid.fit(X_train, y_train)

print(xgb_grid.best_estimator_)
print(xgb_grid.best_params_)
print('[Test] Max ROC_AUC value: {}'.format(xgb_grid.best_score_))
#print('Best std score: {}'.
#      format(xgb_grid.cv_results_['std_test_score'][xgb_grid.best_index_]))
print('[Train] Max ROC_AUC value: {}'.
      format(xgb_grid.cv_results_['mean_train_score'][xgb_grid.best_index_]))
print('Max ROC_AUC value: {}'.format(xgb_grid.best_score_))


# Make a prediction for the test set
y_test = xgb_grid.predict_proba(X_test)[:, 1]

write_to_submission_file(y_test, 'baseline_2_6_xgb.csv')







# Find sessions' starting and ending
full_new_feat['min'] = full_df[times].min(axis=1)
full_new_feat['max'] = full_df[times].max(axis=1)

# Calculate sessions' duration in seconds
full_new_feat['seconds'] = (full_new_feat['max'] - full_new_feat['min']) / np.timedelta64(1, 's')

#full_new_feat['seconds'] = \
#    full_df['time1'].apply(lambda ts: ts.hour).astype('float64')


tmp_scaled = StandardScaler().fit_transform(full_new_feat[['start_month',
                                                           'start_hour',
                                                           'seconds',
                                                           'morning']])
X_train = csr_matrix(hstack([full_sites_sparse[:idx_split,:],
                             tmp_scaled[:idx_split,:]]))
X_test = csr_matrix(hstack([full_sites_sparse[idx_split:,:],
                            tmp_scaled[idx_split:,:]]))

# Capture the quality with default parameters
score_C_ = get_auc_lr_valid(X_train, y_train)
print(score_C_)

Cs = np.logspace(-2, 1, 20)
scores = []
for C in tqdm(Cs):
    scores.append(get_auc_lr_valid(X_train, y_train, C=C))

score_optimal = np.array(scores).max()

plt.plot(Cs, scores, 'ro-')
plt.xscale('log')
plt.xlabel('C')
plt.ylabel('AUC-ROC')
plt.title('Regularization Parameter Tuning')
# horizontal line -- model quality with default C value
plt.axhline(y=score_C_1, linewidth=.5, color='b', linestyle='dashed')
plt.show()


optimal_c = Cs[np.array(scores).argmax()]
round(float(optimal_c), 2)

# Train the model on the whole training data set using optimal regularization parameter
lr2_7 = LogisticRegression(
        C=optimal_c,
        random_state=17,
        solver='liblinear').fit(X_train, y_train)

# Make a prediction for the test set
y_test = lr2_7.predict_proba(X_test)[:, 1]

write_to_submission_file(y_test, 'baseline_2_7.csv')
