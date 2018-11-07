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
from sklearn.preprocessing import StandardScaler, OneHotEncoder

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

experiments_ts = {}



cv = TfidfVectorizer(ngram_range=(1, 3), max_features=50000)
#CountVectorizer(ngram_range=(1, 3), max_features=50000)
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

print(cv_scores)
print('Mean score: {}'.format(cv_scores.mean()))


logit.fit(X_train, y_train)

logit_test_pred = logit.predict_proba(X_test)[:, 1]
write_to_submission_file(logit_test_pred, 'subm1.csv') # 0.91288






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
# Mean score: 0.9124386290425853
# h+sm+m+ev+d+tfidf (0.918710014062 {'C': 0.077426368268112694})
#   ==> 0.94843
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
                start_month,
                morning.values.reshape(-1, 1),
                day.values.reshape(-1, 1),
                evening.values.reshape(-1, 1),
                night.values.reshape(-1, 1),
                week_day
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
# +start_month - month + 1000000  Mean score: 0.9151803425381975
# 0.919913062715 {'C': 0.077426368268112694} => 0.94760
# -------------------------------------------------------------------------
experiment_name = 'ts_tfidf'


def add_features_tfidf2(df,
                        X_sparse,
                        vectorizer,
                        week_day_scaler,
                        start_month_scaler,
                        month_scaler,
                        years_scaler
#                        max_interval_scaler
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

#    month = month_scaler.transform(
#            df['time1'].apply(lambda ts: ts.month).astype('float64').values.reshape(-1, 1))

    years = years_scaler.transform(
            df['time1'].apply(lambda ts: ts.year).astype('int').values.reshape(-1, 1))

#    max_interval = df[times].apply(mean_site_interval, axis=1)

    X = hstack([X_sparse,
                hour.values.reshape(-1, 1),
                start_month,#.values.reshape(-1, 1),
                morning.values.reshape(-1, 1),
                day.values.reshape(-1, 1),
                evening.values.reshape(-1, 1),
                night.values.reshape(-1, 1),
                week_day,
#                month,
                years
                ])

    for site in sites:
        X = hstack([X,
                    vectorizer.transform(df[site].astype('str'))
                    ])

    return X

def mean_site_interval(r):
    mean_v = []
    for i in range(len(times) - 1):
        diff = (r[times[i + 1]] - r[times[i]]) / np.timedelta64(1, 's')
        mean_v.append(diff)

    return np.array(mean_v).mean()


#tt = train_df['time1'].apply(lambda ts: 100 * ts.year + ts.month).astype('float64')
#start_month = StandardScaler().fit_transform(tt.values.reshape(-1,1))
#            train_df['time1'].apply(lambda ts: 100 * ts.year + ts.month).astype('int')

#max_interval_scaler = StandardScaler().fit(
#        train_df[times].apply(mean_site_interval, axis=1).values.reshape(-1,1))


# mean interval
#train_df['mean_interval'] = max_interval_scaler.transform(
#        train_df[times].apply(mean_site_interval, axis=1).values.reshape(-1,1))
#test_df['mean_interval'] = max_interval_scaler.transform(
#        test_df[times].apply(mean_site_interval, axis=1).values.reshape(-1,1))




week_day_scaler = StandardScaler().fit(
        train_df.fillna(0)['time1'].apply(
                lambda ts: ts.weekday()).astype('float64').values.reshape(-1, 1))

start_month_scaler =StandardScaler().fit(
            train_df.fillna(0)['time1'].apply(
                    lambda ts: 100 * ts.year + ts.month).astype('float64').values.reshape(-1, 1))

month_scaler = StandardScaler().fit(
            train_df.fillna(0)['time1'].apply(
                    lambda ts: ts.month).astype('float64').values.reshape(-1, 1))

years_scaler = OneHotEncoder().fit(
            train_df.fillna(0)['time1'].apply(
                    lambda ts: ts.year).astype('int').values.reshape(-1, 1))

# vectorizer TF IDF
vectorizer = TfidfVectorizer(min_df=0.04, max_df=0.8).\
    fit(train_df[sites].fillna(0).astype('str').values.ravel())

#vv = vectorizer.transform(train_df['site1'].fillna(0).astype('str'))

# times & week day _+ TF IDF

X_train_new = add_features_tfidf2(train_df.fillna(0),
                                 X_train,
                                 vectorizer,
                                 week_day_scaler=week_day_scaler,
                                 start_month_scaler=start_month_scaler,
                                 month_scaler=month_scaler,
                                 years_scaler=years_scaler)
X_test_new = add_features_tfidf2(test_df.fillna(0),
                                X_test,
                                vectorizer,
                                week_day_scaler=week_day_scaler,
                                start_month_scaler=start_month_scaler,
                                month_scaler=month_scaler,
                                years_scaler=years_scaler)

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


#  + years => Mean score: 0.9119596767231808
#  => cv: 0.915578414845 {'C': 0.21544346900318834}

#  + years + 1000000 => Mean score: 0.9126369253224578
#  => 0.91580437912 {'C': 0.21544346900318834}

# min_df=0.04, max_df=0.8  Mean score: 0.9152597613178635
# ==> 0.917756935862 {'C': 0.21544346900318834}

# min_df=0.04, max_df=0.8  + mean_interval 0.9152597613178635
#  0.917756935862 {'C': 0.21544346900318834}

# -start_month + month min_df=0.04, max_df=0.8   Mean score: 0.9152597613178635

# +start_month - month + 1000000  Mean score: 0.9151803425381975
# 0.919913062715 {'C': 0.077426368268112694} => 0.94760


print(X_train_new.shape, X_test_new.shape)



logit = LogisticRegression(C=1,
#                           solver='lbfgs',
                           random_state=17)
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




















# -------------------------------------------------------------------------
# h+sm+mor+day+ev+night+weekd Mean score: 0.9154163852457311
# 0.918583948376 {'C': 0.26826957952797248, 'penalty': 'l2'} ==> 0.94283
# -------------------------------------------------------------------------
experiment_name = 'def_ts_week_day_tfidf4'


def add_features_tfidf4(df,
                        X_sparse,
                        vectorizer,
                        weak_day_tranformer,
                        start_month_transformer
                        ):

    hour = df['time1'].apply(lambda ts: ts.hour)
    morning = ((hour >= 7) & (hour <= 11)).astype('int')
    day = ((hour >= 12) & (hour <= 18)).astype('int')
    evening = ((hour >= 19) & (hour <= 23)).astype('int')
    night = ((hour >= 0) & (hour <= 6)).astype('int')

    start_month = start_month_transformer.transform(
            df['time1'].apply(
                    lambda ts: 100 * ts.year + ts.month).astype('float64').values.reshape(-1, 1))

    week_day = weak_day_tranformer.transform(
            df['time1'].apply(lambda ts: ts.weekday()).astype('int').values.reshape(-1, 1))

    n_unique_sites = \
        df[df[sites] != 0][sites].apply(
                lambda site: site[site != 0].nunique(),
                axis=1).astype('float64')

    X = hstack([X_sparse,
                hour.values.reshape(-1, 1),
                start_month,
                morning.values.reshape(-1, 1),
                day.values.reshape(-1, 1),
                evening.values.reshape(-1, 1),
                night.values.reshape(-1, 1),
                week_day
                ])

    for site in sites:
        X = hstack([X,
                    vectorizer.transform(df[site].astype('str'))
                    ])

    return X

def max_site_interval(r):
    mean_v = []
    for i in range(len(times) - 1):
        diff = (r[times[i + 1]] - r[times[i]]) / np.timedelta64(1, 's')
        mean_v.append(diff)

    return np.array(mean_v).max()



# vectorizer TF IDF
vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b').\
    fit(train_df[sites].fillna(0).astype('str').values.ravel())

weak_day_tranformer = OneHotEncoder().fit(
        train_df.fillna(0)['time1'].apply(
                lambda ts: ts.weekday()).astype('int').values.reshape(-1, 1))

start_month_transformer = StandardScaler().fit(
        train_df.fillna(0)['time1'].apply(
                lambda ts: 100 * ts.year + ts.month).astype('float64').values.reshape(-1, 1))

#month_transformer = OneHotEncoder().fit(
#        train_df.fillna(0)['time1'].apply(
#                lambda ts: ts.month).astype('float64').values.reshape(-1, 1))

#ses_len_data = \
#    (train_df.fillna(0)[times].max(axis=1) -
#    train_df.fillna(0)['time1']).astype('timedelta64[s]')
#session_len_transformer = StandardScaler().fit(ses_len_data.values.reshape(-1, 1))

#n_unique_sites_data = \
#    train_df.fillna(0)[train_df.fillna(0)[sites] != 0][sites].apply(
#            lambda site: site[site != 0].nunique(),
#            axis=1).astype('float64')
#n_unique_sites_data_transformer = StandardScaler().fit(
#    n_unique_sites_data.values.reshape(-1, 1))




X_train_new = add_features_tfidf4(train_df.fillna(0),
                                  X_train,
                                  vectorizer,
                                  weak_day_tranformer=weak_day_tranformer,
                                  start_month_transformer=start_month_transformer)
X_test_new = add_features_tfidf4(test_df.fillna(0),
                                 X_test,
                                 vectorizer,
                                 weak_day_tranformer=weak_day_tranformer,
                                 start_month_transformer=start_month_transformer)


# h+sm+mor+day+ev+night+weekd Mean score: 0.9154163852457311
# 0.918583948376 {'C': 0.26826957952797248, 'penalty': 'l2'} ==> 0.94283

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


c_values = np.logspace(-2, 2, 15)

param_grid={'C': c_values,
            'penalty': ['l1', 'l2']},

logit_grid_searcher = GridSearchCV(
        estimator=logit,
        param_grid=param_grid,
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
experiment['submit_file'] = experiment_name + '.csv'

experiments_ts[experiment_name] = experiment


logit_test_pred4 = logit_grid_searcher.predict_proba(X_test_new)[:, 1]
write_to_submission_file(logit_test_pred4, experiment['submit_file'])







# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
experiment_name = 'ts_tfidf3'


def add_features_tfidf3(df,
                        X_sparse,
                        vectorizer,
                        week_day_scaler,
                        start_month_scaler,
                        end_month_scaler,
                        month_scaler,
                        years_scaler
#                        max_interval_scaler
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

    end_month = end_month_scaler.transform(
            df['time10'].apply(
                    lambda ts: 100 * ts.year + ts.month).astype('float64').values.reshape(-1, 1))

    week_day = week_day_scaler.transform(
            df['time1'].apply(lambda ts: ts.weekday()).astype('float64').values.reshape(-1, 1)
            )

    month = month_scaler.transform(
            df['time1'].apply(lambda ts: ts.month).astype('float64').values.reshape(-1, 1))

    years = years_scaler.transform(
            df['time1'].apply(lambda ts: ts.year).astype('int').values.reshape(-1, 1))

#    max_interval = max_interval_scaler.transform(
#            df[times].apply(max_site_interval, axis=1).astype('int').values.reshape(-1, 1))

    X = hstack([X_sparse,
                hour.values.reshape(-1, 1),
                start_month,
                end_month,
                morning.values.reshape(-1, 1),
                day.values.reshape(-1, 1),
                evening.values.reshape(-1, 1),
                night.values.reshape(-1, 1),
                week_day,
                month,
                years
#                max_interval
                ])

    for site in sites:
        X = hstack([X,
                    vectorizer.transform(df[site].astype('str'))
                    ])

    return X

def max_site_interval(r):
    mean_v = []
    for i in range(len(times) - 1):
        diff = (r[times[i + 1]] - r[times[i]]) / np.timedelta64(1, 's')
        mean_v.append(diff)

    return np.array(mean_v).max()


#tt = train_df['time1'].apply(lambda ts: 100 * ts.year + ts.month).astype('float64')
#start_month = StandardScaler().fit_transform(tt.values.reshape(-1,1))
#            train_df['time1'].apply(lambda ts: 100 * ts.year + ts.month).astype('int')

max_interval_scaler = StandardScaler().fit(
        train_df[times].apply(max_site_interval, axis=1).values.reshape(-1,1))


# max interval
train_df['max_interval'] = max_interval_scaler.transform(
        train_df[times].apply(mean_site_interval, axis=1).values.reshape(-1,1))
test_df['max_interval'] = max_interval_scaler.transform(
        test_df[times].apply(mean_site_interval, axis=1).values.reshape(-1,1))




week_day_scaler = StandardScaler().fit(
        train_df.fillna(0)['time1'].apply(
                lambda ts: ts.weekday()).astype('float64').values.reshape(-1, 1))

start_month_scaler =StandardScaler().fit(
            train_df.fillna(0)['time1'].apply(
                    lambda ts: 100 * ts.year + ts.month).astype('float64').values.reshape(-1, 1))

end_month_scaler = StandardScaler().fit(
            train_df.fillna(0)['time10'].apply(
                    lambda ts: 100 * ts.year + ts.month).astype('float64').values.reshape(-1, 1))

month_scaler = StandardScaler().fit(
            train_df.fillna(0)['time1'].apply(
                    lambda ts: ts.month).astype('float64').values.reshape(-1, 1))

years_scaler = OneHotEncoder().fit(
            train_df.fillna(0)['time1'].apply(
                    lambda ts: ts.year).astype('int').values.reshape(-1, 1))

# vectorizer TF IDF
vectorizer = TfidfVectorizer(min_df=0.04, max_df=0.8).\
    fit(train_df[sites].fillna(0).astype('str').values.ravel())

#vv = vectorizer.transform(train_df['site1'].fillna(0).astype('str'))

# times & week day _+ TF IDF

X_train_new = add_features_tfidf3(train_df.fillna(0),
                                 X_train,
                                 vectorizer,
                                 week_day_scaler=week_day_scaler,
                                 start_month_scaler=start_month_scaler,
                                 end_month_scaler=end_month_scaler,
                                 month_scaler=month_scaler,
                                 years_scaler=years_scaler)
X_test_new = add_features_tfidf3(test_df.fillna(0),
                                X_test,
                                vectorizer,
                                week_day_scaler=week_day_scaler,
                                start_month_scaler=start_month_scaler,
                                end_month_scaler=end_month_scaler,
                                month_scaler=month_scaler,
                                years_scaler=years_scaler)

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


#  + years => Mean score: 0.9119596767231808
#  => cv: 0.915578414845 {'C': 0.21544346900318834}

#  + years + 1000000 => Mean score: 0.9126369253224578
#  => 0.91580437912 {'C': 0.21544346900318834}

# min_df=0.04, max_df=0.8  Mean score: 0.9152597613178635
# ==> 0.917756935862 {'C': 0.21544346900318834}

# min_df=0.04, max_df=0.8  + mean_interval 0.9152597613178635
#  0.917756935862 {'C': 0.21544346900318834}

# -start_month + month min_df=0.04, max_df=0.8   Mean score: 0.9152597613178635

# +start_month - month + 1000000  Mean score: 0.9151803425381975
# 0.919913062715 {'C': 0.077426368268112694} => 0.94760


print(X_train_new.shape, X_test_new.shape)



logit = LogisticRegression(C=1,
#                           solver='lbfgs',
                           random_state=17)
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































# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
experiment_name = 'def_ts_week_day_tfidf6'


def add_features_tfidf6(df,
                        X_sparse,
                        vectorizer,
                        weak_day_tranformer,
                        start_month_transformer
                        ):

    hour = df['time1'].apply(lambda ts: ts.hour)
    morning = ((hour >= 7) & (hour <= 11)).astype('int')
    day = ((hour >= 12) & (hour <= 18)).astype('int')
    evening = ((hour >= 19) & (hour <= 23)).astype('int')
    night = ((hour >= 0) & (hour <= 6)).astype('int')

    start_month = start_month_transformer.transform(
            df['time1'].apply(
                    lambda ts: 100 * ts.year + ts.month).astype('float64').values.reshape(-1, 1))

    week_day = weak_day_tranformer.transform(
            df['time1'].apply(lambda ts: ts.weekday()).astype('int').values.reshape(-1, 1))

    n_unique_sites = \
        df[df[sites] != 0][sites].apply(
                lambda site: site[site != 0].nunique(),
                axis=1).astype('float64')

    X = hstack([X_sparse,
                hour.values.reshape(-1, 1),
                start_month,
                morning.values.reshape(-1, 1),
                day.values.reshape(-1, 1),
                evening.values.reshape(-1, 1),
                night.values.reshape(-1, 1),
                week_day
                ])

    for site in sites:
        X = hstack([X,
                    vectorizer.transform(df[site].astype('str'))
                    ])

    return X

def max_site_interval(r):
    mean_v = []
    for i in range(len(times) - 1):
        diff = (r[times[i + 1]] - r[times[i]]) / np.timedelta64(1, 's')
        mean_v.append(diff)

    return np.array(mean_v).max()



# vectorizer TF IDF
vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b').\
    fit(train_df[sites].fillna(0).astype('str').values.ravel())

weak_day_tranformer = OneHotEncoder().fit(
        train_df.fillna(0)['time1'].apply(
                lambda ts: ts.weekday()).astype('int').values.reshape(-1, 1))

start_month_transformer = StandardScaler().fit(
        train_df.fillna(0)['time1'].apply(
                lambda ts: 100 * ts.year + ts.month).astype('float64').values.reshape(-1, 1))

#month_transformer = OneHotEncoder().fit(
#        train_df.fillna(0)['time1'].apply(
#                lambda ts: ts.month).astype('float64').values.reshape(-1, 1))

#ses_len_data = \
#    (train_df.fillna(0)[times].max(axis=1) -
#    train_df.fillna(0)['time1']).astype('timedelta64[s]')
#session_len_transformer = StandardScaler().fit(ses_len_data.values.reshape(-1, 1))

#n_unique_sites_data = \
#    train_df.fillna(0)[train_df.fillna(0)[sites] != 0][sites].apply(
#            lambda site: site[site != 0].nunique(),
#            axis=1).astype('float64')
#n_unique_sites_data_transformer = StandardScaler().fit(
#    n_unique_sites_data.values.reshape(-1, 1))




X_train_new = add_features_tfidf6(train_df.fillna(0),
                                  X_train,
                                  vectorizer,
                                  weak_day_tranformer=weak_day_tranformer,
                                  start_month_transformer=start_month_transformer)
X_test_new = add_features_tfidf6(test_df.fillna(0),
                                 X_test,
                                 vectorizer,
                                 weak_day_tranformer=weak_day_tranformer,
                                 start_month_transformer=start_month_transformer)


# h+sm+mor+day+ev+night+weekd Mean score: 0.9154163852457311
# 0.918583948376 {'C': 0.26826957952797248, 'penalty': 'l2'} ==> 0.94283

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

param_grid={'C': c_values,
            'solver': ['liblinear']},

logit_grid_searcher = GridSearchCV(
        estimator=logit,
        param_grid=param_grid,
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
experiment['submit_file'] = experiment_name + '.csv'

experiments_ts[experiment_name] = experiment


logit_test_pred4 = logit_grid_searcher.predict_proba(X_test_new)[:, 1]
write_to_submission_file(logit_test_pred4, experiment['submit_file'])

























# -------------------------------------------------------------------------
# Mean score: 0.9124386290425853
# h+sm+m+ev+d+tfidf (0.918710014062 {'C': 0.077426368268112694})
#   ==> 0.94843
# -------------------------------------------------------------------------
experiment_name = 'def_ts_week_day_tfidf_no_vec'

def add_features_tfidf7(df_,
                        X_sparse,
                        week_day_scaler,
                        start_month_scaler,
                        end_month_scaler,
                        month_scaler,
                        years_scaler,
                        mean_interval_scaler
                        ):
    df = df_.fillna(0)
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

    time_df = pd.DataFrame(index=df_.index)
    time_df['max'] = df_[times].max(axis=1)
    end_month = end_month_scaler.transform(
            time_df['max'].apply(
                        lambda ts: 100 * ts.year + ts.month).astype('float64').values.reshape(-1, 1))

    week_day = week_day_scaler.transform(
            df['time1'].apply(lambda ts: ts.weekday()).astype('float64').values.reshape(-1, 1)
            )
#
#    month = month_scaler.transform(
#            df['time1'].apply(lambda ts: ts.month).astype('int').values.reshape(-1, 1))
#
#    years = years_scaler.transform(
#                df['time1'].apply(
#                        lambda ts: ts.year).astype('int').values.reshape(-1, 1))
#
#    mean_interval = mean_interval_scaler.transform(
#            df[times].apply(mean_site_interval, axis=1).values.reshape(-1, 1))

    X = hstack([X_sparse,
                hour.values.reshape(-1, 1),
                start_month,
#                end_month,
                morning.values.reshape(-1, 1),
#                day.values.reshape(-1, 1),
#                evening.values.reshape(-1, 1),
#                night.values.reshape(-1, 1),
                week_day
#                month,
#                years,
#                mean_interval
                ])

    return X


def mean_site_interval(r):
    mean_v = [0]
    for i in range(len(times) - 1):
        if (pd.datetime == type(r[times[i + 1]])) and (pd.datetime == type(r[times[i]])):
            diff = (r[times[i + 1]] - r[times[i]]) / np.timedelta64(1, 's')
            mean_v.append(diff)

    return np.array(mean_v).mean()

def max_site_interval(r):
    mean_v = [0]
    for i in range(len(times) - 1):
        if (pd.datetime == type(r[times[i + 1]])) and (pd.datetime == type(r[times[i]])):
            diff = (r[times[i + 1]] - r[times[i]]) / np.timedelta64(1, 's')
            mean_v.append(diff)

    return np.array(mean_v).max()


#tt = train_df['time1'].apply(lambda ts: 100 * ts.year + ts.month).astype('float64')
#start_month = StandardScaler().fit_transform(tt.values.reshape(-1,1))
#            train_df['time1'].apply(lambda ts: 100 * ts.year + ts.month).astype('int')

mean_interval_scaler = StandardScaler().fit(
        train_df[times].apply(mean_site_interval, axis=1).values.reshape(-1,1))
#
#
## max interval
#train_df['mean_interval'] = mean_interval_scaler.transform(
#        train_df[times].apply(mean_site_interval, axis=1).values.reshape(-1,1))
#test_df['mean_interval'] = mean_interval_scaler.transform(
#        test_df[times].apply(mean_site_interval, axis=1).values.reshape(-1,1))
#



week_day_scaler = StandardScaler().fit(
        train_df.fillna(0)['time1'].apply(
                lambda ts: ts.weekday()).astype('float64').values.reshape(-1, 1))

start_month_scaler = StandardScaler().fit(
            train_df.fillna(0)['time1'].apply(
                    lambda ts: 100 * ts.year + ts.month).astype('float64').values.reshape(-1, 1))

end_month_scaler = StandardScaler().fit(
        train_df[times].max(axis=1).apply(
                    lambda ts: 100 * ts.year + ts.month).astype('float64').values.reshape(-1, 1))

month_scaler = OneHotEncoder().fit(
            train_df.fillna(0)['time1'].apply(
                    lambda ts: ts.month).astype('int').values.reshape(-1, 1))

years_scaler = OneHotEncoder().fit(
            train_df.fillna(0)['time1'].apply(
                    lambda ts: ts.year).astype('int').values.reshape(-1, 1))

#vv = vectorizer.transform(train_df['site1'].fillna(0).astype('str'))

# times & week day _+ TF IDF


#end_month = end_month_scaler.transform(
#        train_df[times].max(axis=1).apply(
#                    lambda ts: 100 * ts.year + ts.month).astype('float64').values.reshape(-1, 1))
print(X_train_new.shape, X_test_new.shape)

X_train_new = add_features_tfidf7(train_df,#.fillna(0),
                                 X_train,
                                 week_day_scaler=week_day_scaler,
                                 start_month_scaler=start_month_scaler,
                                 end_month_scaler=end_month_scaler,
                                 month_scaler=month_scaler,
                                 years_scaler=years_scaler,
                                 mean_interval_scaler=mean_interval_scaler)
X_test_new = add_features_tfidf7(test_df,#.fillna(0),
                                X_test,
                                week_day_scaler=week_day_scaler,
                                start_month_scaler=start_month_scaler,
                                end_month_scaler=end_month_scaler,
                                month_scaler=month_scaler,
                                years_scaler=years_scaler,
                                mean_interval_scaler=mean_interval_scaler)

# 7f: Mean score: 0.9086174337377109    0.908670363923 {'C': 0.59948425031894093}
# 7f + month(OneHotEncoder):    Mean score: 0.9059721066961748
# 7f + month(StandardScaler):   Mean score: 0.907251797764966
#               +max_interval:  Mean score: 0.907251041823609
#               +mean_interval: Mean score: 0.907251041823609
#               +end_month:     Mean score: 0.9065307885934872
#                        week_day_OneHot: Mean score: 0.8997776799720713 0.905435146473 {'C': 1.6681005372000592}
# + yaers   Mean score: 0.9061866750468326 0.906776391315 {'C': 1.6681005372000592}
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
experiment['submit_file'] = experiment_name + '.csv'

experiments_ts[experiment_name] = experiment


logit_test_pred4 = logit_grid_searcher.predict_proba(X_test_new)[:, 1]
write_to_submission_file(logit_test_pred4, experiment['submit_file'])



