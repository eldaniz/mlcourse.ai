# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 18:05:48 2018

@author: AEGo
"""

import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
from matplotlib import pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
import pickle
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from tqdm import tqdm

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin


# A helper function for writing predictions to a file
def write_to_submission_file(predicted_labels, out_file,
                             target='target', index_label="session_id"):
    predicted_df = pd.DataFrame(predicted_labels,
                                index = np.arange(1, predicted_labels.shape[0] + 1),
                                columns=[target])
    predicted_df.to_csv(out_file, index_label=index_label)


def add_time_features(df, X_sparse):
    hour = df['time1'].apply(lambda ts: ts.hour)
    morning = ((hour >= 8) & (hour <= 11)).astype('int')
    day = ((hour >= 12) & (hour <= 18)).astype('int')
    evening = ((hour >= 19) & (hour <= 23)).astype('int')
    night = ((hour >= 0) & (hour <= 7)).astype('int')

    weekday = df['time1'].apply(lambda ts: ts.weekday())
#    weekend = (weekday <= 4).astype('int')


    min_d = df[times].min(axis=1)
    max_d = df[times].max(axis=1)

    # Calculate sessions' duration in seconds
    seconds = (max_d - min_d) / np.timedelta64(1, 's')

    n_unique_sites = df[df[sites] != 0][sites].apply(
            lambda site: site[site != 0].nunique(),
            axis=1).astype('float64')

    X = hstack([
            X_sparse,
            morning.values.reshape(-1, 1),
            day.values.reshape(-1, 1),
            evening.values.reshape(-1, 1),
            night.values.reshape(-1, 1),
            seconds.values.reshape(-1, 1),
            n_unique_sites.values.reshape(-1, 1)
#            weekday.values.reshape(-1, 1)
#            weekend.values.reshape(-1, 1)
            ])
    return X


train_df = pd.read_csv('../../data/websites_train_sessions.csv',
                       index_col='session_id')
test_df = pd.read_csv('../../data/websites_test_sessions.csv',
                      index_col='session_id')

# Convert time1, ..., time10 columns to datetime type
times = ['time%s' % i for i in range(1, 11)]
sites = ['site%s' % i for i in range(1, 11)]
train_df[times] = train_df[times].apply(pd.to_datetime)
test_df[times] = test_df[times].apply(pd.to_datetime)

# Sort the data by time
train_df = train_df.sort_values(by='time1')

# Look at the first rows of the training set
train_df.head()


with open('../../data/site_dic.pkl', "rb") as inp_file:
    site_dic = pickle.load(inp_file)

inv_site_dic = {v: k for k, v in site_dic.items()}
# inv_site_dic.update({0: ''})

train_df[sites] = train_df[sites].fillna(0)
test_df[sites] = test_df[sites].fillna(0)

def AAA():
    X_train = train_df[sites].apply(
            lambda x: " ".join(
                    [inv_site_dic[a] for a in x.values if a != 0]), axis=1)

    X_train = X_train.apply(lambda x: x.replace('.', ' '))

    X_test = test_df[sites].apply(
            lambda x: " ".join(
                    [inv_site_dic[a] for a in x.values if a != 0]), axis=1)

    X_test = X_test.apply(lambda x: x.replace('.', ' '))

    y_train = train_df['target'].astype('int')


    pipeline = Pipeline(
            [("vectorize",
              TfidfVectorizer(ngram_range=(1, 3),
                              max_features=100000)),
            ("tfidf", TfidfTransformer())])

    pipeline.fit(X_train.ravel(), y_train)


    X_train = pipeline.transform(X_train.ravel())
    X_test = pipeline.transform(X_test.ravel())


    print(type(X_train))    # scipy.sparse.csr.csr_matrix

    print(X_train.shape)   # (253561, 250000)


    X_train = add_time_features(train_df, X_train)
    X_test = add_time_features(test_df, X_test)


    print(X_train.shape, X_test.shape)  #  ((253561, 250004), (82797, 250004))

    time_split = TimeSeriesSplit(n_splits=12)
    logit = LogisticRegression(C=1, random_state=17)

    # c_values = np.logspace(-2,2, 10)
    # c_values = np.arange(0,5.,step=0.5)
    # c_values = np.concatenate((np.arange(0.1,1,0.1), np.arange(1,5,0.5)))
    c_values = np.concatenate((np.arange(0.5, 2,step=0.5), np.arange(2, 3.6, 0.1)))

    logit_grid_searcher = GridSearchCV(
            estimator=logit,
            param_grid={'C': c_values},
            scoring='roc_auc',
            n_jobs=-1,
            cv=time_split,
            verbose=10)


    logit_grid_searcher.fit(X_train, y_train)


    print(logit_grid_searcher.best_score_, logit_grid_searcher.best_params_)
    # (0.8884620990228279, {'C': 3.5000000000000013})
    # weekend+weekday 0.876261489918 {'C': 3.5000000000000013}
    # weekday  0.876293321626 {'C': 3.5000000000000013}
    # def : 100000   0.890742942071 {'C': 3.5000000000000013}
    #                0.890858325471 {'C': 4.200000000000002}
    #      +seconds+n_unique_sites  0.893673781638 {'C': 3.4000000000000012}

    logit_test_pred = logit_grid_searcher.predict_proba(X_test)[:, 1]
    write_to_submission_file(logit_test_pred, 'submit.csv')

































# Special transformer to save output shape
class ShapeSaver(BaseEstimator, TransformerMixin):
    def transform(self, X):
        self.shape = X.shape
        return X

    def fit(self, X, y=None, **fit_params):
        return self

#####################################
## Helper functions that extract different data
#####################################

# Return sites columns as a single string
# This string can be supplied into CountVectorizer or TfidfVectorizer

def extract_sites_as_string(X):
    #return X[sites].astype('str').apply(' '.join, axis=1)
    return X['sites_str']



# Year-month feature from A4
def feature_year_month(X):
    return pd.DataFrame(X['time1'].dt.year * 100 + X['time1'].dt.month)

def feature_year_month_log1p(X):
    return pd.DataFrame(np.log1p(X['time1'].dt.year * 100 + X['time1'].dt.month))

# yearfeature from A4
def feature_year(X):
    return pd.DataFrame(X['time1'].dt.year)

# Hour feature from A4
def feature_hour(X):
    return pd.DataFrame(X['time1'].dt.hour)

# Hour feature from A4
def feature_hour_log(X):
    return np.log1p(pd.DataFrame(X['time1'].dt.hour))


# Month
def feature_month(X):
    return pd.DataFrame(X['time1'].dt.month)

# Weekday
def feature_weekday(X):
    return pd.DataFrame(X['time1'].dt.weekday)

# Is day feature from A4
def feature_is_daytime(X):
    return pd.DataFrame( (X['time1'].dt.hour >= 12) & (X['time1'].dt.hour <= 18))

# Is evening feature from A4
def feature_is_evening(X):
    return pd.DataFrame( (X['time1'].dt.hour >= 19) & (X['time1'].dt.hour <= 23))

# Is morning feature from A4
def feature_is_morning(X):
    return pd.DataFrame(X['time1'].dt.hour <= 11)

# Long Session length feature from A4
def feature_is_long_session(X):
    X['session_end_time'] = X[times].max(axis=1)
    session_duration = (X['session_end_time'] - X['time1']).astype('timedelta64[s]')
#    q = session_duration.quantile([0.1, 0.90]).values
    X['long_session_duration'] = 0
    X[session_duration < 10]['long_session_duration'] = 1
    X[session_duration < 20]['long_session_duration'] = 2
    X[session_duration < 100]['long_session_duration'] = 3
    X[session_duration < 500]['long_session_duration'] = 4
    X[session_duration < 1000]['long_session_duration'] = 5
#    X[(session_duration > q[1]) & (session_duration <= q[2])]['long_session_duration'] = 2
#    X[(session_duration > q[2]) & (session_duration <= q[3])]['long_session_duration'] = 3
#    X[(session_duration > q[3]) & (session_duration <= q[4])]['long_session_duration'] = 4
#    X[session_duration > q[1]]['long_session_duration'] = 2
    return X[['long_session_duration']]

# Session length feature from A4
def feature_session_len(X):
    X['session_end_time'] = X[times].max(axis=1)
    X['session_duration'] = (X['session_end_time'] - X['time1']).astype('timedelta64[s]')
    return X[['session_duration']]

# uniq sites per session
def feature_uniq_sites(X):
    X['n_unique_sites'] = X[X[sites] != 0][sites].apply(
            lambda site: site[site != 0].nunique(), axis=1).astype('float64')

    return X[['n_unique_sites']]


transform_pipeline = Pipeline([
    ('features', FeatureUnion([
        # List of features goes here:
#        ('year_month_val', Pipeline([
#            ('extract', FunctionTransformer(feature_year_month, validate=False)),
#            ('scale', StandardScaler()),
#            ('shape', ShapeSaver())
#        ])),
        ('session_len', Pipeline([
            ('extract', FunctionTransformer(feature_session_len, validate=False)),
            ('scale', StandardScaler()),
            ('shape', ShapeSaver())
        ])),
        ('weekday_cat', Pipeline([
            ('extract', FunctionTransformer(feature_weekday, validate=False)),
            ('ohe', OneHotEncoder()),
            ('shape', ShapeSaver())
        ])),
#        ('hour_val', Pipeline([
#            ('extract', FunctionTransformer(feature_hour, validate=False)),
##            ('scale', StandardScaler()),
#            ('ohe', OneHotEncoder()),
#            ('shape', ShapeSaver())
#         ])),
        ('hour_val_log1p', Pipeline([
            ('extract', FunctionTransformer(feature_hour_log, validate=False)),
            ('scale', StandardScaler()),
            ('shape', ShapeSaver())
         ])),
        ('hour_cat', Pipeline([
            ('extract', FunctionTransformer(feature_hour, validate=False)),
            ('ohe', OneHotEncoder()),
            ('shape', ShapeSaver())
         ])),
        ('month_cat', Pipeline([
            ('extract', FunctionTransformer(feature_month, validate=False)),
            ('ohe', OneHotEncoder()),
            ('shape', ShapeSaver())
         ])),
        ('is_morning', Pipeline([
            ('extract', FunctionTransformer(feature_is_morning, validate=False)),
            ('shape', ShapeSaver())
         ])),
        ('is_daytime', Pipeline([
            ('extract', FunctionTransformer(feature_is_daytime, validate=False)),
            ('shape', ShapeSaver())
         ])),
        ('is_evening', Pipeline([
            ('extract', FunctionTransformer(feature_is_evening, validate=False)),
            ('shape', ShapeSaver())
         ])),
        ('is_long_session', Pipeline([
            ('extract', FunctionTransformer(feature_is_long_session, validate=False)),
            ('ohe', OneHotEncoder()),
            ('shape', ShapeSaver())
         ])),
#        ('feature_uniq_sites', Pipeline([
#            ('extract', FunctionTransformer(feature_uniq_sites, validate=False)),
#            ('ohe', OneHotEncoder()),
#            ('shape', ShapeSaver())
#         ])),
        ('year', Pipeline([
            ('extract', FunctionTransformer(feature_year, validate=False)),
            ('ohe', OneHotEncoder()),
            ('shape', ShapeSaver())
         ])),
        ('sites_tfidf', Pipeline([
            ('extract', FunctionTransformer(extract_sites_as_string, validate=False)),
            ('count', TfidfVectorizer(token_pattern=r'(?u)\b\w+\b',
                                      ngram_range=(1, 3),
                                      max_features=100000)),
            ("tfidf", TfidfTransformer()),
            ('shape', ShapeSaver())
        ])),
        # Add more features here :)
        # ...
    ]))
])


# Run preprocessing on full data
x_train_new = train_df.iloc[:, :-1]
x_train_new['sites_str'] = train_df[sites].apply(
        lambda x: " ".join(
                [inv_site_dic[a] for a in x.values if a != 0]), axis=1)

x_train_new['sites_str'] = x_train_new['sites_str'].apply(lambda x: x.replace('.', ' '))

x_test_new = test_df.iloc[:, :]
x_test_new['sites_str'] = test_df[sites].apply(
        lambda x: " ".join(
                [inv_site_dic[a] for a in x.values if a != 0]), axis=1)

x_test_new['sites_str'] = x_test_new['sites_str'].apply(lambda x: x.replace('.', ' '))

transformed_train_df = transform_pipeline.fit_transform(x_train_new)
transformed_test_df = transform_pipeline.transform(x_test_new)

X_train_new = transformed_train_df
y_train_new = train_df['target']

print(transformed_train_df.shape, transformed_test_df.shape)

time_split = TimeSeriesSplit(n_splits=12)

logit = LogisticRegression(C=1, random_state=17)

# c_values = np.logspace(-2,2, 10)
# c_values = np.arange(0,5.,step=0.5)
# c_values = np.concatenate((np.arange(0.1,1,0.1), np.arange(1,5,0.5)))
c_values = np.concatenate((np.arange(0.5, 2,step=0.5), np.arange(2, 3.6, 0.1)))


clf = LogisticRegression(C=1, random_state=17)  # RandomForestClassifier(random_state=17)

tree_params = {
        'max_depth': [2, 5, 10],
        'max_features': [3, 5, 20]}

#c_values = np.logspace(-4, 10, 40)
logit_params = {
        'C': [0.1, 0.05, 0.15, 2.0309, 3.5],
        'solver': ['lbfgs']#, 'sag', 'saga'],
#        'penalty' : ['l1', 'l2']
        },

clf_grid_searcher = GridSearchCV(
        estimator=clf,
        param_grid=logit_params,
        scoring='roc_auc',
        n_jobs=-1,
        cv=time_split,
        verbose=10)


clf_grid_searcher.fit(X_train_new, y_train_new)
print(clf_grid_searcher.score(X_train_new, y_train_new))
print(clf_grid_searcher.best_score_, clf_grid_searcher.best_params_)

# -- RandomForestClassifier :
#all : 0.868859388644 {'max_depth': 2, 'max_features': 3}
#- day_time - eve -long_sess: 0.866209364039 {'max_depth': 2, 'max_features': 5} \
#               --    -sess_len    0.85821693632 {'max_depth': 10, 'max_features': 5} \
# - year_month  0.881829021147 {'max_depth': 5, 'max_features': 10} / !!!!!!
# - week_day    0.863236779933 {'max_depth': 2, 'max_features': 5}  \
# - hour_Val    0.837083132261 {'max_depth': 10, 'max_features': 3}  \
# - hour_cat    0.859024053379 {'max_depth': 5, 'max_features': 3}  \
# - mon_cat     0.871291662775 {'max_depth': 10, 'max_features': 3}   / !!!!!!
# - is_morning  0.867803370795 {'max_depth': 10, 'max_features': 5} \
# - is_daytime  0.85062178331 {'max_depth': 5, 'max_features': 3}  \
# - is_evening  0.865505228744 {'max_depth': 10, 'max_features': 5}  \
# - is_long_ses 0.868926633714 {'max_depth': 10, 'max_features': 5}  / !!!!


# -- LogisticRegression :
#all : 0.840667160623 {'C': 1.0}
#- day_time - eve -long_sess:
#               --    -sess_len
# - year_month     0.856734589393 {'C': 1}  / !!!!!
# - week_day       0.838524546676 {'C': 0.15} \
# - hour_Val       0.840135383489 {'C': 1} \ ~=
# - hour_cat       0.78535407395 {'C': 0.15} \\\
# + hour_val(ohe): 0.842485256959 {'C': 1} / !!!!
# - hour_cat, + hour_val(ohe) 0.840135383489 {'C': 1} \
# - mon_cat        0.832168325842 {'C': 0.05}  \
# - is_morning     0.838739921881 {'C': 1}   \
# - is_daytime     0.84074001553 {'C': 1}  / !!!!
# - is_evening     0.840012789159 {'C': 1}  \
# - is_long_ses    0.835454317425 {'C': 1}  \
# all 5quantiles  : 0.840667160623 {'C': 1.0} -
# all 2quantiles  : 0.840667160623 {'C': 1} -
# all 2quantiles (0.1, 0.9)  : 0.840667160623 {'C': 1} -
# +uniq_sites(ohe):0.844042953365 {'C': 1}  / !!!!
# +uniq_sites(sc): 0.840681126851 {'C': 1} / !
# +uniq_sites + hour_val(ohe) (+hour_cat): 0.84578279885 {'C': 1} / !!!! ???????????
# -hour_cat +uniq_sites + hour_val(ohe):  0.843802942673 {'C': 1} / !!!
# +hour_cat +uniq_sites - hour_val(ohe):  0.843802942673 {'C': 1} / !!!
# -- +year :        0.845713208798 {'C': 1} / !!!!!
# -hour +log1p(hour)     0.845895327182 {'C': 1} / !!!!!
#   -- -hour_cat         0.786349908391 {'C': 0.15} \ ---
#   -- -year_mon         0.854384041716 {'C': 1} / !!!
#0.993002559728
#0.909865994457 {'C': 2.0309176209047348, 'random_state': 17, 'solver': 'lbfgs'}
# 0.910397293314 {'C': 2.0309, 'solver': 'lbfgs'}



# + Tfidf:          0.907521078033 {'C': 3.5}
# + Tfidf+ transformer:  0.90336907447 {'C': 3.5}
#                        0.903462223502 {'C': 4.6415888336127775}
#  -hour +log1p(hour) +year -year_month  0.908940245543 {'C': 3.5} (0.90840662684 {'C': 3.5})
#      --      n_iniq :                 -0.85375820649 {'C': 1}


clf = LogisticRegression(
        random_state=17,
        **clf_grid_searcher.best_params_)
clf.fit(X_train_new, y_train_new)
clf.score(X_train_new, y_train_new)
# 0.99544488308533252



#transform_pipeline.steps[0][1].transformer_list[2][1].steps[1][1]
#feature_names = [f[0] for f in transform_pipeline.steps[0][1].transformer_list]
#feat_importances = pd.Series(clf.feature_importances_, index=feature_names)
#feat_importances.nlargest(15).plot(kind='barh')


# (0.8884620990228279, {'C': 3.5000000000000013})
# weekend+weekday 0.876261489918 {'C': 3.5000000000000013}
# weekday  0.876293321626 {'C': 3.5000000000000013}
# def : 100000   0.890742942071 {'C': 3.5000000000000013}
#                0.890858325471 {'C': 4.200000000000002}
#      +seconds+n_unique_sites  0.893673781638 {'C': 3.4000000000000012}

logit_test_pred = clf.predict_proba(transformed_test_df)[:, 1]
write_to_submission_file(logit_test_pred, 'no_ym_fg.csv')








