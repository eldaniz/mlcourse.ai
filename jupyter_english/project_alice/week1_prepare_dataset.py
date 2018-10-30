# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 18:10:43 2018

@author: AEGo
"""

# Disable Anaconda warnings
import warnings
warnings.filterwarnings('ignore')
from glob import glob
import os
import pickle
# pip install tqdm
from tqdm import tqdm_notebook
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

import glob
from tqdm import tqdm

# Change the path to data
PATH_TO_DATA = '../../data/capstone_user_identification'

user31_data = pd.read_csv(os.path.join(PATH_TO_DATA,  '10users/user0031.csv'))

user31_data.head()

session_length = 10
# sites columns
columns = ['site%s' % i for i in range(1, session_length+1)]
# target: user_id
columns.append('user_id')
sites_columns = ['site%s' % i for i in range(1, session_length+1)]

def prepare_train_set(path_to_csv_files, session_length=10):
    """
    Parameters
    ----------
    path_to_csv_files : str
        path to the directory containing csv-files
    session_length : str
        number of websites in a session

    Returns
    -------
    two objects
        1. DataFrame with rows corresponding to a unique user session
        (each of session_length websites) and session_length columns
        corresponding to the websites in the session. And there
        should be one more column - user_id which is target class
        of a sample. Hence the DataFrame should have session_length + 1
        columns.
        2. Dictionary of websites' frequncies which has
        {'site_string': [site_id, site_freq]} structure,
        where site_string - original website name,
        site_id - a number the website got be encoded,
        site_freq - total number of website occurrences in all files
        among all users. For example:
            {'vk.com': (1, 2),
            'google.com': (2, 2),
            'yandex.ru': (3, 3),
            'facebook.com': (4, 1)}
    """

    # sites columns
    columns = ['site%s' % i for i in range(1, session_length+1)]
    # target: user_id
    columns.append('user_id')
    sites_columns = ['site%s' % i for i in range(1, session_length+1)]

    # result DataFrame
    df = pd.DataFrame(columns=columns)
    # result Dictionary
    sites = {}

    for next_file in tqdm(glob.glob(os.path.join(path_to_csv_files, '*.csv'))):
        # get user id: userNNNN.csv
        user_id = int(next_file[-8:-4])
        orig_file_df = pd.read_csv(next_file)

        full_sessions_count = orig_file_df.shape[0] // session_length

        if full_sessions_count > 0:
            full_sess_df = \
                orig_file_df[:(session_length*full_sessions_count)]['site'].\
                values.flatten().reshape(full_sessions_count, session_length)
            file_df = pd.DataFrame(full_sess_df, columns=sites_columns)
            file_df['user_id'] = user_id

        last_session_part = orig_file_df.shape[0] % session_length

        if last_session_part > 0:
            parted = pd.DataFrame(columns=columns)
            parted = pd.DataFrame(data=[[''] * (session_length+1)],
                                        columns=columns)
            parted[sites_columns] = ''
            parted['user_id'] = user_id
            parted.iloc[:, :last_session_part] = \
                orig_file_df[-last_session_part:]['site'].values.flatten()
            if full_sessions_count > 0:
                file_df = pd.concat([file_df, parted], ignore_index=True)
            else:
                file_df = parted

        df = pd.concat([df, file_df], ignore_index=True)

    stat = pd.DataFrame(data=df[sites_columns].values.flatten(),
                        columns=['site'])
    values = stat['site'].value_counts(sort=True)
    stat = pd.DataFrame(data=list(zip(values.index.values, values.values)),
                        columns=['site', 'count'])
    empte_pos = stat.loc[stat['site'] == '']

    if not empte_pos.empty:
        empte_pos = empte_pos.index[0]
        empty_value = stat.iloc[empte_pos].copy()
        stat.drop(empte_pos, inplace=True)
        # need 'empty'?
        stat = stat.append(empty_value)

    sites = {site: (site_id+1, site_freq)
        for site, site_freq, site_id in zip(stat['site'].values,
                                            stat['count'].values,
                                            range(stat.shape[0]))}

    # empty?
    if sites.get(''):
        sites[''] = (0, sites[''][1])

    df[sites_columns] = df[sites_columns].apply(lambda sess: [sites.get(key)[0] for key in sess.values])

    del sites['']

    return df, sites

#!cat $PATH_TO_DATA/3users/user0001.csv

df, sites = prepare_train_set(os.path.join(PATH_TO_DATA, '3users'), session_length=10)
sites


#stat = pd.DataFrame(data=df[sites_columns].values.flatten(),
#                        columns=['site'])
#ordered_values = stat['site'].value_counts(sort=True)
#
#stat['count'] = stat['site'].apply(lambda site: ordered_values[site])
#
#sorted(list(zip(stat['site'].values, stat['count'].values)),
#       key=lambda s: s[1],
#       reverse=True)


#df.index = np.arange(1, df.index.shape[0] + 1)
#sites_columns = ['site%s' % i for i in range(1, 10+1)]
#stat = pd.DataFrame(data=df[sites_columns].values.flatten(), columns=['site'])
#values = stat['site'].value_counts(sort=True)
#stat = pd.DataFrame(data=list(zip(values.index.values, values.values)), columns=['site', 'count'])
#empte_pos=stat.loc[stat['site']=='']
#if not empte_pos.empty:
#    empte_pos=empte_pos.index[0]
#    empty_value = stat.iloc[empte_pos].copy()
#    stat.drop(empte_pos, inplace=True)
#    stat = stat.append(empty_value)
#stat
#
#sites = { site: (site_id+1, site_freq) for site, site_freq, site_id in zip(stat['site'].values, stat['count'].values, range(stat.shape[0])) }



#sites = { k: [i + 1,v] for k, v, i in zip(stat.index, stat.values, range(stat.count())) }



#sites = {}
#def func(site):
#    func.cur_id = 1
#    print(site)
#    if not sites.get(site):
#        if site == '':
#            sites[''] = [0, 0]
#        else:
#            sites[site] = [ func.cur_id, 0]
#            ++func.cur_id
#
#    sites[site][1] = sites[site][1] + 1
#    return sites[site][0]
#
#sites = {}
#def func_ttl(site1):
#    #print(site)
#    #print(site1[:1])
#    site = site1
##    print(type(site1))
##    print(site1)
#
#    if not sites.get(site):
#        if site == '':
#            sites[''] = [0, 0]
#        else:
#            cur_id = func_ttl.cur_id
#            func_ttl.cur_id = func_ttl.cur_id + 1
#            sites[site] = [ cur_id, 0]
#            ++cur_id
#
#    sites[site][1] = sites[site][1] + 1
#    return site, sites[site][0]
#func_ttl.cur_id = 1
#
#
##stat['site'].apply(func)
#stat['site'].apply(func_ttl)
#sites
#sites['vk.com'][1]
#
##aa = df
##sites_columns = ['site%s' % i for i in range(1, 2+1)]
##stat = pd.DataFrame(data=df[sites_columns].values.flatten(), columns=['site'])
##
##
##stat.iloc[0], stat.iloc[1] = c,b
##
##stat.index[1] [stat=='']
##as_list = stat.index.tolist()
##idx = as_list.index('')
##np.array(as_list) + 1
##as_list[idx] = 'South Korea'
##stat.index = as_list
##
##stat = stat['site'].value_counts(sort=False)
##sites = { k: [i + 1,v] for k, v, i in zip(stat.index, stat.values, range(stat.count())) }
#
##df[sites_columns] = df[sites_columns].apply(lambda sess: [sites.get(key)[0] for key in sess.values])
#
##ss =sites.copy()

#columns = ['site%s' % i for i in range(1, 10+1)]
#d = [ [ i for i in range(10) ]]
##d = [ [1,2,3,4,5,6,7,8,9,10]  ]
#parted = pd.DataFrame(data = d, columns=columns)
#pd.concat([pd.DataFrame([ [i] ] , columns=columns) for i in range(10)],
#               ignore_index=True)
#


train_data_toy, site_freq_3users = \
    prepare_train_set(os.path.join(PATH_TO_DATA, '3users'),
                      session_length=10)

# -----------------------------------------------------------------------
# Question 1.
#  How many unique sessions with length of 10 websites are there in 10users data?

train_data_10users, site_freq_10users = \
    prepare_train_set(os.path.join(PATH_TO_DATA, '10users'), session_length=10)
print(train_data_10users.iloc[:, :-1].count()[0])  # 14061
#print(train_data_10users.iloc[:, :-1].drop_duplicates().count()[0])  # 13183

# -----------------------------------------------------------------------
# Questioin 2.
# How many unique websites are there in 10users data?

print(len(site_freq_10users))  # 4913


# -----------------------------------------------------------------------
# Question 3.
# How many unique sessions with length of 10 websites are there in 150users data?

train_data_150users, site_freq_150users = \
    prepare_train_set(os.path.join(PATH_TO_DATA, '150users'),
                      session_length=10)
print(train_data_150users.iloc[:, :-1].count()[0])  # 137019
#print(train_data_150users.iloc[:, :-1].drop_duplicates().count()[0])  # 131009

# -----------------------------------------------------------------------
# Question 4.
# How many unique websites are there in 150users data?

print(len(site_freq_150users))  # 27797


# -----------------------------------------------------------------------
# Question 5.
# Which of these websites is NOT in top-10 most visited websites among 150 users?
#
#    www.google.fr
#    www.youtube.com
#    safebrowsing-cache.google.com
#    www.linkedin.com

search_set = {'www.google.fr', 'www.youtube.com',
              'safebrowsing-cache.google.com', 'www.linkedin.com'}
# [[k, site_freq_150users[k]] for k in list(site_freq_150users)[:10]]
top10_sites = {k for k in list(site_freq_150users)[:10]}
print(search_set - top10_sites)  # {'www.linkedin.com'}


train_data_10users.to_csv(
        os.path.join(PATH_TO_DATA, 'train_data_10users.csv'),
        index_label='session_id', float_format='%d')

train_data_150users.to_csv(
        os.path.join(PATH_TO_DATA, 'train_data_150users.csv'),
        index_label='session_id',
        float_format='%d')

X_toy, y_toy = \
    train_data_toy.iloc[:, :-1].values, train_data_toy.iloc[:, -1].values

print(X_toy)

X_toy_flatten = X_toy.flatten()
X_sparse_toy = csr_matrix(
    (
        [1] * X_toy_flatten.shape[0],
        X_toy_flatten,
        range(0, X_toy_flatten.shape[0] + 10, 10)
    )
    )[:, 1:]

X_sparse_toy.todense()

X_10users, y_10users = train_data_10users.iloc[:, :-1].values, \
                       train_data_10users.iloc[:, -1].values
X_150users, y_150users = train_data_150users.iloc[:, :-1].values, \
                         train_data_150users.iloc[:, -1].values

train_10users_no_target_df = train_data_10users[sites_columns]
train_data_10users_flatten = train_10users_no_target_df.values.flatten()
X_sparse_10users = csr_matrix(
    (
        [1] * train_data_10users_flatten.shape[0],
        train_data_10users_flatten,
        range(0, train_data_10users_flatten.shape[0] + session_length,
              session_length)
    )
)[:, 1:]

train_150users_no_target_df = train_data_150users[sites_columns]
train_data_150users_flatten = train_150users_no_target_df.values.flatten()
X_sparse_150users = csr_matrix(
    (
        [1] * train_data_150users_flatten.shape[0],
        train_data_150users_flatten,
        range(0, train_data_150users_flatten.shape[0] + session_length,
              session_length)
    )
)[:, 1:]

# Save these sparse matrices with pickle (serialization in Python),
# and y_10users, y_150users - target variables (user_id-s) for our 10users
# and 150users data. The fact that the names of these matrices start
# with X and y implies that we are going to check our first classification
# models on them. Finally, save frequency dictionaries for 3users, 10users
# and 150users data.

with open(os.path.join(PATH_TO_DATA, 'X_sparse_10users.pkl'), 'wb') as X10_pkl:
    pickle.dump(X_sparse_10users, X10_pkl, protocol=2)
with open(os.path.join(PATH_TO_DATA, 'y_10users.pkl'), 'wb') as y10_pkl:
    pickle.dump(y_10users, y10_pkl, protocol=2)
with open(os.path.join(PATH_TO_DATA,
                       'X_sparse_150users.pkl'), 'wb') as X150_pkl:
    pickle.dump(X_sparse_150users, X150_pkl, protocol=2)
with open(os.path.join(PATH_TO_DATA, 'y_150users.pkl'), 'wb') as y150_pkl:
    pickle.dump(y_150users, y150_pkl, protocol=2)
with open(os.path.join(PATH_TO_DATA,
                       'site_freq_3users.pkl'), 'wb') as site_freq_3users_pkl:
    pickle.dump(site_freq_3users, site_freq_3users_pkl, protocol=2)
with open(
        os.path.join(
                PATH_TO_DATA,
                'site_freq_10users.pkl'), 'wb') as site_freq_10users_pkl:
    pickle.dump(site_freq_10users, site_freq_10users_pkl, protocol=2)
with open(
        os.path.join(
                PATH_TO_DATA,
                'site_freq_150users.pkl'), 'wb') as site_freq_150users_pkl:
    pickle.dump(site_freq_150users, site_freq_150users_pkl, protocol=2)


# Just in case doublecheck that number of columns in sparse matrices
# X_sparse_10users and X_sparse_150users equals to the number of unique
# websites in 10users and 150users data evaluated earlier.
assert X_sparse_10users.shape[1] == len(site_freq_10users)
assert X_sparse_150users.shape[1] == len(site_freq_150users)

