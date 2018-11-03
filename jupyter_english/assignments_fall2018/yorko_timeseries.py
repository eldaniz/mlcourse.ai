# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 18:57:42 2018

@author: AEGo
"""

# Import libraries and set desired options
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

# A helper function for writing predictions to a file
def write_to_submission_file(predicted_labels, out_file,
                             target='target', index_label="session_id"):
    predicted_df = pd.DataFrame(predicted_labels,
                                index = np.arange(1, predicted_labels.shape[0] + 1),
                                columns=[target])
    predicted_df.to_csv(out_file, index_label=index_label)

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
print(train_df.head())


sites = ['site%s' % i for i in range(1, 11)]
train_df[sites].fillna(0).astype('int').to_csv('train_sessions_text.txt',
                                               sep=' ',
                       index=None, header=None)
test_df[sites].fillna(0).astype('int').to_csv('test_sessions_text.txt',
                                              sep=' ',
                       index=None, header=None)




cv = CountVectorizer(ngram_range=(1, 3), max_features=50000)
with open('train_sessions_text.txt') as inp_train_file:
    X_train = cv.fit_transform(inp_train_file)
with open('test_sessions_text.txt') as inp_test_file:
    X_test = cv.transform(inp_test_file)
print(X_train.shape, X_test.shape)


y_train = train_df['target'].astype('int')



time_split = TimeSeriesSplit(n_splits=10)

[(el[0].shape, el[1].shape) for el in time_split.split(X_train)]


logit = LogisticRegression(C=1, random_state=17)


cv_scores = cross_val_score(
        logit, X_train,
        y_train,
        cv=time_split,
        scoring='roc_auc',
        n_jobs=-1)

print(cv_scores, cv_scores.mean())


logit.fit(X_train, y_train)

logit_test_pred = logit.predict_proba(X_test)[:, 1]
write_to_submission_file(logit_test_pred, 'subm1.csv') # 0.91288



experiments_ts = {}


# -------------------------------------------------------------------------
# ts 0.916380752986 {'C': 0.59948425031894093} + week_day ==> 0.94696
# -------------------------------------------------------------------------
experiment_name = 'def_ts_week_day'


def add_features(df, X_sparse):
    hour = df['time1'].apply(lambda ts: ts.hour)
    morning = ((hour >= 7) & (hour <= 11)).astype('int')
    day = ((hour >= 12) & (hour <= 18)).astype('int')
    evening = ((hour >= 19) & (hour <= 23)).astype('int')
    night = ((hour >= 0) & (hour <= 6)).astype('int')

    week_day = df['time1'].apply(lambda ts: ts.weekday()).astype('int')

    X = hstack([X_sparse,
                morning.values.reshape(-1, 1),
                day.values.reshape(-1, 1),
                evening.values.reshape(-1, 1),
                night.values.reshape(-1, 1),
                week_day.values.reshape(-1, 1)
                ])
    return X


# times & week day
X_train_new = add_features(train_df.fillna(0), X_train)
X_test_new = add_features(test_df.fillna(0), X_test)
print(X_train_new.shape, X_test_new.shape)


cv_scores = cross_val_score(logit,
                            X_train_new,
                            y_train,
                            cv=time_split,
                            scoring='roc_auc',
                            n_jobs=-1)

print(cv_scores, cv_scores.mean())  # 0.91559979 ???


logit.fit(X_train_new, y_train)
#LogisticRegression(C=1, class_weight=None, dual=False, fit_intercept=True,
#          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
#          penalty='l2', random_state=17, solver='liblinear', tol=0.0001,
#          verbose=0, warm_start=False)


logit_test_pred2 = logit.predict_proba(X_test_new)[:, 1]
write_to_submission_file(logit_test_pred2, 'subm2.csv')  # 0.93843


c_values = np.logspace(-2, 2, 10)

logit_grid_searcher = GridSearchCV(
        estimator=logit,
        param_grid={'C': c_values},
        scoring='roc_auc',
        n_jobs=-1,
        cv=time_split,
        verbose=1)


logit_grid_searcher.fit(X_train_new, y_train)


# -week_day: (0.9173763958236849, {'C': 0.21544346900318834}) ==> 0.94242
# +week_day: 0.916380752986 {'C': 0.59948425031894093} ==> 0.94696
print(logit_grid_searcher.best_score_,
      logit_grid_searcher.best_params_)


experiment = {}
experiment['logit_grid_searcher.best_score_'] = logit_grid_searcher.best_score_
experiment['logit_grid_searcher.best_params_'] = logit_grid_searcher.best_params_
experiment['cv_scores.best_params_'] = cv_scores
experiment['cv_scores.mean'] = cv_scores.mean()
experiment['submit_file'] = 'subm3.csv'

experiments_ts[experiment_name] = experiment


logit_test_pred3 = logit_grid_searcher.predict_proba(X_test_new)[:, 1]
write_to_submission_file(logit_test_pred3, experiment['submit_file'])  # weekday=0.94696 (def:0.94242)

# -------------------------------------------------------------------------
# h+sm+m+ev+d+tfidf (0.918710014062 {'C': 0.077426368268112694}) Mean score: 0.9124386290425853
# ==> 0.94843
# -------------------------------------------------------------------------
experiment_name = 'def_ts_week_day_tfidf'


def add_features_tfidf(df, X_sparse, vectorizer):
    hour = df['time1'].apply(lambda ts: ts.hour)
    morning = ((hour >= 7) & (hour <= 11)).astype('int')
    day = ((hour >= 12) & (hour <= 18)).astype('int')
    evening = ((hour >= 19) & (hour <= 23)).astype('int')
    night = ((hour >= 0) & (hour <= 6)).astype('int')

    start_month = StandardScaler().fit_transform(
            df['time1'].apply(
                    lambda ts: 100 * ts.year + ts.month).astype('float64').values.reshape(-1, 1))
    week_day = StandardScaler().fit_transform(
            df['time1'].apply(lambda ts: ts.weekday()).astype('float64').values.reshape(-1, 1))

    X = hstack([X_sparse,
                hour.values.reshape(-1, 1),
                start_month,#.values.reshape(-1, 1),
                morning.values.reshape(-1, 1),
                day.values.reshape(-1, 1),
                evening.values.reshape(-1, 1),
                night.values.reshape(-1, 1),
                week_day#.values.reshape(-1, 1)
                ])

    for site in sites:
        X = hstack([X,
                    vectorizer.transform(df[site].astype('str'))
                    ])

    return X


# vectorizer TF IDF
vectorizer = TfidfVectorizer().\
    fit(train_df[sites].fillna(0).astype('str').values.ravel())


X_train_new = add_features_tfidf(train_df.fillna(0), X_train, vectorizer)
X_test_new = add_features_tfidf(test_df.fillna(0), X_test, vectorizer)

# def:                      0.9117890204757252
# min_df=3, max_df=0.3:     0.9117890204757252
# min_df=0.1, max_df=0.9:   0.913996846420335
# min_df=0.03, max_df=0.8:  0.9150225892335152
# min_df=0.03, max_df=0.5:  0.9150225892335152
# min_df=0.03:              0.9150225892335152
# min_df=0.05, max_df=0.8:  error
# min_df=0.04, max_df=0.8   0.9152263080988643 (grid_cv: 0.916001944198 {'C': 0.59948425031894093})
# max_df=0.3:               0.9117890204757252
# min_df=0.02:              0.9138372280401713 (0.915780079137 {'C': 0.59948425031894093})

print(X_train_new.shape, X_test_new.shape)


logit = LogisticRegression(C=1, random_state=17)
cv_scores = cross_val_score(logit,
                            X_train_new,
                            y_train,
                            cv=time_split,
                            scoring='roc_auc',
                            n_jobs=-1)

print(cv_scores)
print('Mean score: {}'.format(cv_scores.mean()))


c_values = np.logspace(-2, 2, 10)

logit_grid_searcher = GridSearchCV(
        estimator=logit,
        param_grid={'C': c_values},
        scoring='roc_auc',
        n_jobs=-1,
        cv=time_split,
        verbose=1)

logit_grid_searcher.fit(X_train_new, y_train)


print(logit_grid_searcher.best_score_,
      logit_grid_searcher.best_params_)


experiment = {}
experiment['logit_grid_searcher.best_score_'] = logit_grid_searcher.best_score_
experiment['logit_grid_searcher.best_params_'] = logit_grid_searcher.best_params_
experiment['cv_scores.best_params_'] = cv_scores
experiment['cv_scores.mean'] = cv_scores.mean()
experiment['submit_file'] = 'subm4.csv'

experiments_ts[experiment_name] = experiment


logit_test_pred4 = logit_grid_searcher.predict_proba(X_test_new)[:, 1]
write_to_submission_file(logit_test_pred4, experiment['submit_file'])





# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
experiment_name = 'ts_tfidf'


def add_features_tfidf2(df,
                        X_sparse,
                        vectorizer,
                        week_day_scaler,
                        start_month_scaler
                        ):
    hour = df['time1'].apply(lambda ts: ts.hour)
    morning = ((hour >= 7) & (hour <= 11)).astype('int')
    day = ((hour >= 12) & (hour <= 18)).astype('int')
    evening = ((hour >= 19) & (hour <= 23)).astype('int')
    night = ((hour >= 0) & (hour <= 6)).astype('int')

#weekday = df['time1'].apply(lambda ts: ts.dayofweek)
#df['weekday'] = df['time1'].apply(lambda ts: ts.dayofweek)
#df['weekend'] = ((weekday >= 5) & (weekday <= 6)).astype('int')
#df['weekdays'] = (weekday <= 4).astype('int')
#    years = df['time1'].apply(lambda ts: ts.year)
    start_month = start_month_scaler.transform(
            df['time1'].apply(
                    lambda ts: 100 * ts.year + ts.month).astype('float64').values.reshape(-1, 1))

    week_day = week_day_scaler.transform(
            df['time1'].apply(lambda ts: ts.weekday()).astype('float64').values.reshape(-1, 1)
            )

    X = hstack([X_sparse,
                hour.values.reshape(-1, 1),
#                years.values.reshape(-1, 1),
                start_month,#.values.reshape(-1, 1),
                morning.values.reshape(-1, 1),
                day.values.reshape(-1, 1),
                evening.values.reshape(-1, 1),
                night.values.reshape(-1, 1),
                week_day#.values.reshape(-1, 1)
                ])

    for site in sites:
        X = hstack([X,
                    vectorizer.transform(df[site].astype('str'))
                    ])

    return X

#tt = train_df['time1'].apply(lambda ts: 100 * ts.year + ts.month).astype('float64')
#start_month = StandardScaler().fit_transform(tt.values.reshape(-1,1))
#            train_df['time1'].apply(lambda ts: 100 * ts.year + ts.month).astype('int')

week_day_scaler = StandardScaler().fit(
        train_df.fillna(0)['time1'].apply(
                lambda ts: ts.weekday()).astype('float64').values.reshape(-1, 1))

start_month_scaler =StandardScaler().fit(
            train_df.fillna(0)['time1'].apply(
                    lambda ts: 100 * ts.year + ts.month).astype('float64').values.reshape(-1, 1))
# vectorizer TF IDF
vectorizer = TfidfVectorizer().\
    fit(train_df[sites].fillna(0).astype('str').values.ravel())

#vv = vectorizer.transform(train_df['site1'].fillna(0).astype('str'))

# times & week day _+ TF IDF

X_train_new = add_features_tfidf2(train_df.fillna(0),
                                 X_train,
                                 vectorizer,
                                 week_day_scaler=week_day_scaler,
                                 start_month_scaler=start_month_scaler)
X_test_new = add_features_tfidf2(test_df.fillna(0),
                                X_test,
                                vectorizer,
                                week_day_scaler=week_day_scaler,
                                start_month_scaler=start_month_scaler)

# def:                      0.9117890204757252
# min_df=3, max_df=0.3:     0.9117890204757252
# min_df=0.1, max_df=0.9:   0.913996846420335
# min_df=0.03, max_df=0.8:  0.9150225892335152
# min_df=0.03, max_df=0.5:  0.9150225892335152
# min_df=0.03:              0.9150225892335152
# min_df=0.05, max_df=0.8:  error
# min_df=0.04, max_df=0.8   0.9152263080988643 (grid_cv: 0.916001944198 {'C': 0.59948425031894093})
# max_df=0.3:               0.9117890204757252
# min_df=0.02:              0.9138372280401713 (0.915780079137 {'C': 0.59948425031894093})

print(X_train_new.shape, X_test_new.shape)


logit = LogisticRegression(C=1, random_state=17)
cv_scores = cross_val_score(logit,
                            X_train_new,
                            y_train,
                            cv=time_split,
                            scoring='roc_auc',
                            n_jobs=-1)

print(cv_scores)
print('Mean score: {}'.format(cv_scores.mean()))


c_values = np.logspace(-2, 2, 10)

logit_grid_searcher = GridSearchCV(
        estimator=logit,
        param_grid={'C': c_values},
        scoring='roc_auc',
        n_jobs=-1,
        cv=time_split,
        verbose=1)

logit_grid_searcher.fit(X_train_new, y_train)


print(logit_grid_searcher.best_score_,
      logit_grid_searcher.best_params_)


experiment = {}
experiment['logit_grid_searcher.best_score_'] = logit_grid_searcher.best_score_
experiment['logit_grid_searcher.best_params_'] = logit_grid_searcher.best_params_
experiment['cv_scores.best_params_'] = cv_scores
experiment['cv_scores.mean'] = cv_scores.mean()
experiment['submit_file'] = experiment_name+ '.csv'

experiments_ts[experiment_name] = experiment


logit_test_pred4 = logit_grid_searcher.predict_proba(X_test_new)[:, 1]
write_to_submission_file(logit_test_pred4, experiment['submit_file'])

