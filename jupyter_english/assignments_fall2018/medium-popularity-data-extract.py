
# coding: utf-8

# # Introduction:
# What is Medium? Medium is a dynamically developing international publishing platform for people to write, read and clap easily online. It is like the russian [habrahabr.ru](http://habrahabr.ru) just a little worse. We have two JSON files that contain published articles on Medium till 2018, March. There is number of claps to each article in the first file and there is no ones in the second file. Our goal is to predict the number of "claps" for articles in test.
# Let's start our EDA journey!

# [ ]:


import numpy as np
import pandas as pd
import json
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from scipy.sparse import csr_matrix, hstack
from scipy.stats import probplot
import pickle
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import gc
import warnings
warnings.filterwarnings('ignore')
import time
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.linear_model import Ridge



color = sns.color_palette()
sns.set_style("whitegrid")
sns.set_context("paper")
sns.palplot(color)

import os
PATH = "../../data"


# [ ]:


get_ipython().system('du -l ../../data/*')


# # 1. Data preprocessing
# ## 1.1. Supplementary functions

# [ ]:


def read_json_line(line=None):
    result = None
    try:
        result = json.loads(line)
    except Exception as e:
        # Find the offending character index:
        idx_to_replace = int(str(e).split(' ')[-1].replace(')',''))
        # Remove the offending character:
        new_line = list(line)
        new_line[idx_to_replace] = ' '
        new_line = ''.join(new_line)
        return read_json_line(line=new_line)
    return result

from html.parser import HTMLParser

class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()

def extract_features(path_to_data):

    content_list = []
    published_list = []
    title_list = []
    author_list = []
    domain_list = []
    tags_list = []
    url_list = []

    with open(path_to_data, encoding='utf-8') as inp_json_file:
        for line in inp_json_file:
            json_data = read_json_line(line)
#             content = json_data['content'].replace('\n', ' ').replace('\r', ' ') # ORIG
            content = json_data['content'].replace('\n', ' \n ').replace('\r', ' \n ') # keep newline
            content_no_html_tags = strip_tags(content)
            content_list.append(content_no_html_tags)
            published = json_data['published']['$date']
            published_list.append(published)
            title = json_data['meta_tags']['title'].split('\u2013')[0].strip() #'Medium Terms of Service – Medium Policy – Medium'
            title_list.append(title)
            author = json_data['meta_tags']['author'].strip()
            author_list.append(author)
            domain = json_data['domain']
            domain_list.append(domain)
            url = json_data['url']
            url_list.append(url)

            tags_str = []
            soup = BeautifulSoup(content, 'lxml')
            try:
                tag_block = soup.find('ul', class_='tags')
                tags = tag_block.find_all('a')
                for tag in tags:
                    tags_str.append(tag.text.translate({ord(' '):None, ord('-'):None}))
                tags = ' '.join(tags_str)
            except Exception:
                tags = 'None'
            tags_list.append(tags)

    return content_list, published_list, title_list, author_list, domain_list, tags_list, url_list


# ## 1.2. Data extraction

# [ ]:


content_list, published_list, title_list, author_list, domain_list, tags_list, url_list = \
    extract_features(os.path.join(PATH, 'train.json'))

train = pd.DataFrame()
train['content'] = content_list
train['published'] = pd.to_datetime(published_list, format='%Y-%m-%dT%H:%M:%S.%fZ')
train['title'] = title_list
train['author'] = author_list
train['domain'] = domain_list
train['tags'] = tags_list
# train['length'] = train['content'].apply(len)
train['url'] = url_list

content_list, published_list, title_list, author_list, domain_list, tags_list, url_list = \
    extract_features(os.path.join(PATH, 'test.json'))

test = pd.DataFrame()
test['content'] = content_list
test['published'] = pd.to_datetime(published_list, format='%Y-%m-%dT%H:%M:%S.%fZ')
test['title'] = title_list
test['author'] = author_list
test['domain'] = domain_list
test['tags'] = tags_list
# test['length'] = test['content'].apply(len)
test['url'] = url_list


# [ ]:


del content_list, published_list, title_list, author_list, domain_list, tags_list, url_list
gc.collect()


# [ ]:


train['target'] = pd.read_csv(os.path.join(
        PATH,
        'train_log1p_recommends.csv'),
    index_col='id').values


# [ ]:


train.tail()


# [ ]:


train.describe()


# [ ]:


train.to_csv("mediumPopularity.csv.gz", index=False, compression="gzip")
train_df = train

# [ ]:


test.to_csv("mediumPopularity_test.csv.gz", index=False, compression="gzip")
test_df = test


# --------------------------------------------------------------------------
def write_submission_file(prediction,
                          filename,
                          path_to_sample=os.path.join(
                                  PATH,
                                  'sample_submission.csv')):
    submission = pd.read_csv(path_to_sample, index_col='id')

    submission['log_recommends'] = prediction
    submission.to_csv(filename)


# --------------------------------------------------------------------------
# Special transformer to save output shape
class ShapeSaver(BaseEstimator, TransformerMixin):
    def transform(self, X):
        self.shape = X.shape
        return X

    def fit(self, X, y=None, **fit_params):
        return self


#--------------------------------------------------------------------------
# Run preprocessing on full data
x_train_new = train_df.iloc[:, :-1]
x_test_new = test_df.iloc[:, :].copy()
experiments = {}


#####################################
# Helper functions that extract different data
#####################################


# Return sites columns as a single string
# This string can be supplied into CountVectorizer or TfidfVectorizer

def extract_content_as_string(X):
    return X['content']


def extract_author_as_string(X):
    return X['author']


def extract_tags_as_string(X):
    return X['tags']

#--------------------------------------------------------------------------
transform_pipeline = Pipeline([
    ('features', FeatureUnion([
        # List of features goes here:
#        ('author_tfidf', Pipeline([
#            ('extract', FunctionTransformer(extract_author_as_string, validate=False)),
#            ('count', TfidfVectorizer(max_features=5000)),
##            ("tfidf", TfidfTransformer()),
#            ('shape', ShapeSaver())
#        ])),

#        ('tags_tfidf', Pipeline([
#            ('extract', FunctionTransformer(extract_tags_as_string, validate=False)),
#            ('count', TfidfVectorizer(max_features=25000)),
##            ("tfidf", TfidfTransformer()),
#            ('shape', ShapeSaver())
#        ])),

        ('content_tfidf', Pipeline([
            ('extract', FunctionTransformer(extract_content_as_string, validate=False)),
            ('count', TfidfVectorizer(max_features=50000)),
#            ("tfidf", TfidfTransformer()),
            ('shape', ShapeSaver())
        ])),
        # Add more features here :)
        # ...
    ]))
])


#--------------------------------------------------------------------------
# experiments
#--------------------------------------------------------------------------
#{'med_baseline_1': {'submission_file': 'med_baseline_1.csv',
#  'transformed_train_df_shape': (62313, 100000),
#  'transformed_test_df_shape': (34645, 100000),
#  'clf': 'ridge',
#  'valid_mae': 1.1864877347713028,
#  'np.expm1_valid_mae': 2.2755563573670612},
# 'med_2': {'submission_file': 'med_2.csv',
#  'transformed_train_df_shape': (62313, 100000),
#  'transformed_test_df_shape': (34645, 100000),
#  'clf': 'ridge',
#  'valid_mae': 1.1787153133497037,
#  'np.expm1_valid_mae': 2.250196036188286}}
#
# only content (50000)
#{'submission_file': 'med_2.csv',
# 'transformed_train_df_shape': (62313, 50000),
# 'transformed_test_df_shape': (34645, 50000),
# 'clf': 'ridge',
# 'valid_mae': 1.185198115758509,
# 'np.expm1_valid_mae': 2.2713348602576606}
#

# only content (100000)
#{'med_baseline_1': {'submission_file': 'med_baseline_1.csv',
#  'transformed_train_df_shape': (62313, 100000),
#  'transformed_test_df_shape': (34645, 100000),
#  'clf': 'ridge',
#  'valid_mae': 1.1864877347713028,
#  'np.expm1_valid_mae': 2.2755563573670612},

# content+author (50000 + 5000)
# 'med_2': {'submission_file': 'med_2.csv',
#  'transformed_train_df_shape': (62313, 55000),
#  'transformed_test_df_shape': (34645, 55000),
#  'clf': 'ridge',
#  'valid_mae': 1.1517136446320084,
#  'np.expm1_valid_mae': 2.1636095698060336}}

#--------------------------------------------------------------------------


#--------------------------------------------------------------------------
#{'submission_file': 'med_baseline_1.csv',
# 'transformed_train_df_shape': (62313, 50000),
# 'transformed_test_df_shape': (34645, 50000),
# 'clf': 'ridge',
# 'valid_mae': 1.1945453278258193,
# 'np.expm1_valid_mae': 2.3020560761228199}
# ==> 1.80926 corr => 1.53729
#--------------------------------------------------------------------------
def med_baselin1_1():
    experiment_name = 'med_baseline_1'
    experiment = {}
    experiment['submission_file'] = experiment_name + '.csv'

    transformed_train_df = transform_pipeline.fit_transform(x_train_new)
    transformed_test_df = transform_pipeline.transform(x_test_new)

    X_train_new = transformed_train_df
    y_train_new = train_df['target']
    X_test_new = transformed_test_df

    print(transformed_train_df.shape, transformed_test_df.shape)

    experiment['transformed_train_df_shape'] = transformed_train_df.shape
    experiment['transformed_test_df_shape'] = transformed_test_df.shape


    train_part_size = int(0.7 * y_train_new.shape[0])
    X_train_part = X_train_new[:train_part_size, :]
    y_train_part = y_train_new[:train_part_size]
    X_valid =  X_train_new[train_part_size:, :]
    y_valid = y_train_new[train_part_size:]

    ridge = Ridge(random_state=17)
    ridge.fit(X_train_part, y_train_part);
    ridge_pred = ridge.predict(X_valid)

    plt.hist(y_valid, bins=30, alpha=.5, color='red', label='true', range=(0,10));
    plt.hist(ridge_pred, bins=30, alpha=.5, color='green', label='pred', range=(0,10));
    plt.legend();

    valid_mae = mean_absolute_error(y_valid, ridge_pred)
    print(valid_mae, np.expm1(valid_mae))
    experiment['clf'] = 'ridge'
    experiment['valid_mae'] = valid_mae
    experiment['np.expm1_valid_mae'] = np.expm1(valid_mae)

    ridge.fit(X_train_new, y_train_new)
    ridge_test_pred = ridge.predict(X_test_new)

    ridge_test_pred_corrected = ridge_test_pred + (all_zero_mae - ridge_test_pred.mean())
    print(ridge_test_pred_corrected.mean(), all_zero_mae)

    write_submission_file(prediction=ridge_test_pred_corrected,
                          filename=experiment['submission_file'])

#    write_submission_file(prediction=ridge_test_pred,
#                          filename=experiment['submission_file'])
    experiments[experiment_name] = experiment

    with open('medium_experiments.pickle', 'wb') as f:
        pickle.dump(experiments, f)






    # All Zeroes ==> 4.33328
    write_submission_file(np.zeros_like(ridge_test_pred),
                          'medium_all_zeros_submission.csv')


all_zero_mae = 4.33328



#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
experiment_name = 'med_2'
experiment = {}
experiment['submission_file'] = experiment_name + '.csv'
experiment['experiment_time'] = time.asctime()

transformed_train_df = transform_pipeline.fit_transform(x_train_new)
transformed_test_df = transform_pipeline.transform(x_test_new)

X_train_new = transformed_train_df
y_train_new = train_df['target']
X_test_new = transformed_test_df

print(transformed_train_df.shape, transformed_test_df.shape)

experiment['transformed_train_df_shape'] = transformed_train_df.shape
experiment['transformed_test_df_shape'] = transformed_test_df.shape
experiment['features'] = [v[0] for v in transform_pipeline.steps[0][1].transformer_list]

train_part_size = int(0.7 * y_train_new.shape[0])
X_train_part = X_train_new[:train_part_size, :]
y_train_part = y_train_new[:train_part_size]
X_valid =  X_train_new[train_part_size:, :]
y_valid = y_train_new[train_part_size:]


ridge = Ridge(random_state=17)
ridge.fit(X_train_part, np.log1p(y_train_part));
ridge_pred = np.expm1(ridge.predict(X_valid))


plt.hist(y_valid, bins=30, alpha=.5, color='red', label='true', range=(0,10));
plt.hist(ridge_pred, bins=30, alpha=.5, color='green', label='pred', range=(0,10));
plt.legend();

valid_mae = mean_absolute_error(y_valid, ridge_pred)
print(valid_mae, np.expm1(valid_mae))
experiment['clf'] = 'ridge'
experiment['valid_mae'] = valid_mae
experiment['np.expm1_valid_mae'] = np.expm1(valid_mae)
experiments[experiment_name] = experiment

with open('medium_experiments.pickle', 'wb') as f:
    pickle.dump(experiments, f)

ridge.fit(X_train_new, y_train_new)
ridge_test_pred = ridge.predict(X_test_new)

ridge_test_pred_corrected = ridge_test_pred + (all_zero_mae - ridge_test_pred.mean())
print(ridge_test_pred_corrected.mean(), all_zero_mae)

write_submission_file(prediction=ridge_test_pred_corrected,
                      filename=experiment['submission_file'])
















