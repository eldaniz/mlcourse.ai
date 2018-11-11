
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
from nltk.stem import PorterStemmer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingRegressor

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

def init():
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


    train.to_csv("mediumPopularity.csv.gz", index=False, compression="gzip")
    test.to_csv("mediumPopularity_test.csv.gz", index=False, compression="gzip")
# [ ]:


#train.to_csv("mediumPopularity.csv.gz", index=False, compression="gzip")
#train = pd.read_csv("mediumPopularity.csv.gz", compression="gzip")
train_df = train


# [ ]:


#
#test = pd.read_csv("mediumPopularity_test.csv.gz", compression="gzip")
test_df = test


all_zero_mae = 4.33328


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


# --------------------------------------------------------------------------
# Run preprocessing on full data
x_train_new = train_df.iloc[:, :-1]
x_test_new = test_df.iloc[:, :].copy()
experiments = {}

with open('medium_experiments.pickle', 'rb') as f:
    experiments = pickle.load(f)



# train_df['title'].value_counts()
# train_df['published'].describe()
# train_df['published'].count()

#####################################
# Helper functions that extract different data
#####################################

stemmer = PorterStemmer()

# Return sites columns as a single string
# This string can be supplied into CountVectorizer or TfidfVectorizer

def extract_content_as_string(X):
    return X['content']


def extract_content_length(X):
    return pd.DataFrame(X['content'].str.len())


def extract_author_as_string(X):
    return X['author']


def extract_tags_as_string(X):
    return X['tags']


def extract_title_as_string(X):
    return X['title']

def extract_domain_as_string(X):
    return X['domain']


def feature_weekday(X):
    return pd.DataFrame(X['published'].dt.weekday)


def feature_hour(X):
    return pd.DataFrame(X['published'].dt.hour)


def feature_month(X):
    return pd.DataFrame(X['published'].dt.month)


# yearfeature from A4
def feature_year(X):
    return pd.DataFrame(X['published'].dt.year)


# Month Q
def feature_month_q1(X):
    return pd.DataFrame(X['published'].dt.month.isin([1, 2, 3]))


# Month Q
def feature_month_q2(X):
    return pd.DataFrame(X['published'].dt.month.isin([4, 5, 6]))


# Month Q
def feature_month_q3(X):
    return pd.DataFrame(X['published'].dt.month.isin([7, 8, 9]))


# Month Q
def feature_month_q4(X):
    return pd.DataFrame(X['published'].dt.month.isin([10, 11, 12]))


def stem_tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = []
    for item in tokens:
        stems.append(PorterStemmer().stem(item))
    return stems


class StemmedCountVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])


# --------------------------------------------------------------------------
transform_pipeline = Pipeline([
    ('features', FeatureUnion([
        # List of features goes here:
        ('author_tfidf', Pipeline([
            ('extract', FunctionTransformer(extract_author_as_string, validate=False)),
            ('count', TfidfVectorizer(ngram_range=(2, 2), max_features=10000)),
#            ("tfidf", TfidfTransformer()),
            ('shape', ShapeSaver())
        ])),
#
        ('domain_tfidf', Pipeline([
            ('extract', FunctionTransformer(extract_domain_as_string, validate=False)),
            ('count', TfidfVectorizer(max_features=10000)),
#            ("tfidf", TfidfTransformer()),
            ('shape', ShapeSaver())
        ])),

#        ('weekday_cat', Pipeline([
#            ('extract', FunctionTransformer(feature_weekday, validate=False)),
#            ('ohe', OneHotEncoder()),
#            ('shape', ShapeSaver())
#        ])),

#        ('month_cat', Pipeline([
#            ('extract', FunctionTransformer(feature_month, validate=False)),
#            ('ohe', OneHotEncoder()),
#            ('shape', ShapeSaver())
#         ])),

#        ('hour_val', Pipeline([
#            ('extract', FunctionTransformer(feature_hour, validate=False)),
##            ('scale', StandardScaler()),
#            ('ohe', OneHotEncoder()),
#            ('shape', ShapeSaver())
#         ])),

#        ('mon_q1', Pipeline([
#            ('extract', FunctionTransformer(feature_month_q1, validate=False)),
#            ('shape', ShapeSaver())
#         ])),
#        ('mon_q2', Pipeline([
#            ('extract', FunctionTransformer(feature_month_q2, validate=False)),
#            ('shape', ShapeSaver())
#         ])),
#        ('mon_q3', Pipeline([
#            ('extract', FunctionTransformer(feature_month_q3, validate=False)),
#            ('shape', ShapeSaver())
#         ])),
#        ('mon_q4', Pipeline([
#            ('extract', FunctionTransformer(feature_month_q4, validate=False)),
#            ('shape', ShapeSaver())
#         ])),

#        ('year', Pipeline([
#            ('extract', FunctionTransformer(feature_year, validate=False)),
#            ('ohe', OneHotEncoder()),
#            ('shape', ShapeSaver())
#         ])),

#        ('tags_tfidf', Pipeline([
#            ('extract', FunctionTransformer(extract_tags_as_string, validate=False)),
#            ('count', TfidfVectorizer(max_features=10000)),
##            ("tfidf", TfidfTransformer()),
#            ('shape', ShapeSaver())
#        ])),
#
#        ('title_tfidf', Pipeline([
#            ('extract', FunctionTransformer(extract_title_as_string, validate=False)),
#            ('count', TfidfVectorizer(max_features=10000)),
##            ("tfidf", TfidfTransformer()),
#            ('shape', ShapeSaver())
#        ])),

#        ('content_length', Pipeline([
#            ('extract', FunctionTransformer(extract_content_length, validate=False)),
#            ('scale', StandardScaler()),
#            ('shape', ShapeSaver())
#        ])),

        ('content_tfidf', Pipeline([
            ('extract', FunctionTransformer(extract_content_as_string, validate=False)),
            ('count', TfidfVectorizer(max_features=50000)),
#            ("tfidf", TfidfTransformer()),
            ('shape', ShapeSaver())
        ])),

#    ('content13_tfidf', Pipeline([
#            ('extract', FunctionTransformer(extract_content_as_string, validate=False)),
#            ('count', TfidfVectorizer(ngram_range=(1, 3), max_features=50000)),
##            ("tfidf", TfidfTransformer()),
#            ('shape', ShapeSaver())
#        ])),

#        ('content_stem_tfidf', Pipeline([
#            ('extract', FunctionTransformer(extract_content_as_string, validate=False)),
#            ('count', StemmedCountVectorizer(max_features=50000)),
##            ("tfidf", TfidfTransformer()),
#            ('shape', ShapeSaver())
#        ])),

#        ('content_stem_tokenize_tfidf', Pipeline([
#            ('extract', FunctionTransformer(extract_content_as_string, validate=False)),
#            ('count', TfidfVectorizer(tokenizer=stem_tokenize,
#                                             max_features=50000)),
##            ("tfidf", TfidfTransformer()),
#            ('shape', ShapeSaver())
##        ])),
        # Add more features here :)
        # ...
    ]))
])


# --------------------------------------------------------------------------
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
#  '09_11_2018_15:06:00': {'submission_file': 'med_2.csv',
#  'experiment_time': '09_11_2018_15:05:08',
#  'transformed_train_df_shape': (62313, 50000),
#  'transformed_test_df_shape': (34645, 50000),
#  'features': ['content_tfidf'],
#  'clf': 'ridge',
#  'valid_mae': 1.185198115758509,
#  'np.expm1_valid_mae': 2.2713348602576606}}
#            +Tfidftransformer : 1.20297773249 2.33001807693
#    + content13_tfidf ngram_range=(1, 3) 1.1774090096 2.24595306484

#{'time': '10_11_2018_15_29_20',
# 'submission_file': '10_11_2018_15_29_20.csv',
# 'transformed_train_df_shape': (62313, 50000),
# 'transformed_test_df_shape': (34645, 50000),
# 'features': ['content13_tfidf'],
# 'clf': 'ridge',
# 'valid_mae': 1.1707294140789162,
# 'np.expm1_valid_mae': 2.224343662652017}

# only content (100000)
# {'med_baseline_1': {'submission_file': 'med_baseline_1.csv',
#  'transformed_train_df_shape': (62313, 100000),
#  'transformed_test_df_shape': (34645, 100000),
#  'clf': 'ridge',
#  'valid_mae': 1.1864877347713028,
#  'np.expm1_valid_mae': 2.2755563573670612},
#
#
# content+author (50000 + 5000)
# 'med_2': {'submission_file': 'med_2.csv',
#  'transformed_train_df_shape': (62313, 55000),
#  'transformed_test_df_shape': (34645, 55000),
#  'clf': 'ridge',
#  'valid_mae': 1.1517136446320084,
#  'np.expm1_valid_mae': 2.1636095698060336}}

# content+tag(50000 + 25000)
#     '09_11_2018_15:13:28': {'submission_file': 'med_2.csv',
#  'transformed_train_df_shape': (62313, 75000),
#  'transformed_test_df_shape': (34645, 75000),
#  'features': ['tags_tfidf', 'content_tfidf'],
#  'clf': 'ridge',
#  'valid_mae': 1.1702598469703198,
#  'np.expm1_valid_mae': 2.2228299723386296}}
#
# content+tag(50000 + 5000)
#  '09_11_2018_15:18:16': {'submission_file': 'med_2.csv',
#  'transformed_train_df_shape': (62313, 55000),
#  'transformed_test_df_shape': (34645, 55000),
#  'features': ['tags_tfidf', 'content_tfidf'],
#  'clf': 'ridge',
#  'valid_mae': 1.1753514810292236,
#  'np.expm1_valid_mae': 2.2392812896944014}}
#
# content+author+tag(50000 + 5000 + 25000)
# '09_11_2018_15:23:35': {'submission_file': 'med_2.csv',
#  'transformed_train_df_shape': (62313, 80000),
#  'transformed_test_df_shape': (34645, 80000),
#  'features': ['author_tfidf', 'tags_tfidf', 'content_tfidf'],
#  'clf': 'ridge',
#  'valid_mae': 1.1455553783608081,
#  'np.expm1_valid_mae': 2.1441870854939511}}

# content+title(50000 + 10000)
# '09_11_2018_15:31:55': {'submission_file': 'med_2.csv',
#  'transformed_train_df_shape': (62313, 60000),
#  'transformed_test_df_shape': (34645, 60000),
#  'features': ['title_tfidf', 'content_tfidf'],
#  'clf': 'ridge',
#  'valid_mae': 1.2156653138903533,
#  'np.expm1_valid_mae': 2.3725371138604436}}   \ !!!

# content+title(50000 + 1000)
# '09_11_2018_21:42:14': {'submission_file': 'med_2.csv',
#  'transformed_train_df_shape': (62313, 51000),
#  'transformed_test_df_shape': (34645, 51000),
#  'features': ['title_tfidf', 'content_tfidf'],
#  'clf': 'ridge',
#  'valid_mae': 1.1857798394872872,
#  'np.expm1_valid_mae': 2.2732384269919401}}

# only content (100000) (ngram=1,3)
#{'time': '09_11_2018_22_44_05',
# 'submission_file': '09_11_2018_22_44_05.csv',
# 'transformed_train_df_shape': (62313, 100000),
# 'transformed_test_df_shape': (34645, 100000),
# 'features': ['content_tfidf'],
# 'clf': 'ridge',
# 'valid_mae': 1.1545836555155469,
# 'np.expm1_valid_mae': 2.1727022054429548}

# content+tag+author(50000 + 25000 + 5000) no log1p
#{'time': '10_11_2018_11_28_05',
# 'submission_file': '10_11_2018_11_28_05.csv',
# 'transformed_train_df_shape': (62313, 80000),
# 'transformed_test_df_shape': (34645, 80000),
# 'features': ['author_tfidf', 'tags_tfidf', 'content_tfidf'],
# 'clf': 'ridge',
# 'valid_mae': 1.1618052611035263,
# 'np.expm1_valid_mae': 2.1956971396563754}
#    log1p!!!
#{'time': '10_11_2018_11_33_49',
# 'submission_file': '10_11_2018_11_33_49.csv',
# 'features': ['author_tfidf', 'tags_tfidf', 'content_tfidf'],
# 'transformed_train_df_shape': (62313, 80000),
# 'transformed_test_df_shape': (34645, 80000),
# 'clf': 'ridge',
# 'valid_mae': 1.1455553783608081,
# 'np.expm1_valid_mae': 2.1441870854939511}

# content+tag+author(50000 + 25000 + 5000)+scale(content_len) log1p
#{'time': '10_11_2018_11_40_15',
# 'submission_file': '10_11_2018_11_40_15.csv',
# 'transformed_train_df_shape': (62313, 80001),
# 'transformed_test_df_shape': (34645, 80001),
# 'features': ['author_tfidf', 'tags_tfidf', 'content_length', 'content_tfidf'],
# 'clf': 'ridge',
# 'valid_mae': 1.7790883889164533,
# 'np.expm1_valid_mae': 4.9244531589659895}

# content+tag+author(50000 + 25000 + 5000)+no_scale(content_len) log1p
#{'time': '10_11_2018_11_47_17',
# 'submission_file': '10_11_2018_11_47_17.csv',
# 'transformed_train_df_shape': (62313, 80001),
# 'transformed_test_df_shape': (34645, 80001),
# 'features': ['author_tfidf', 'tags_tfidf', 'content_length', 'content_tfidf'],
# 'clf': 'ridge',
# 'valid_mae': 1.8278513033648709,
# 'np.expm1_valid_mae': 5.2205063075433094}


#        ('content_stem_tfidf', Pipeline([
#            ('extract', FunctionTransformer(extract_content_as_string, validate=False)),
#            ('count', StemmedCountVectorizer(max_features=50000)),
##            ("tfidf", TfidfTransformer()),
#            ('shape', ShapeSaver())
#        ])),
#    {'time': '10_11_2018_12_05_05',
# 'submission_file': '10_11_2018_12_05_05.csv',
# 'transformed_train_df_shape': (62313, 50000),
# 'transformed_test_df_shape': (34645, 50000),
# 'features': ['content_stem_tfidf'],
# 'clf': 'ridge',
# 'valid_mae': 1.791537194607818,
# 'np.expm1_valid_mae': 4.9986665004856174}

# content (50000) + author_tfidf( (2,2) +5000)
#{'time': '10_11_2018_18_13_00',
# 'submission_file': '10_11_2018_18_13_00.csv',
# 'transformed_train_df_shape': (62313, 55000),
# 'transformed_test_df_shape': (34645, 55000),
# 'features': ['author_tfidf', 'content_tfidf'],
# 'clf': 'ridge',
# 'valid_mae': 1.1130807934110891,
# 'np.expm1_valid_mae': 2.043721040568097}

# content (50000) + author_tfidf( (2,2) +10000)
#{'time': '10_11_2018_18_24_58',
# 'submission_file': '10_11_2018_18_24_58.csv',
# 'transformed_train_df_shape': (62313, 60000),
# 'transformed_test_df_shape': (34645, 60000),
# 'features': ['author_tfidf', 'content_tfidf'],
# 'clf': 'ridge',
# 'valid_mae':                                            1.1047306784298387, base
# 'np.expm1_valid_mae': 2.0184114360205081}
#           alpha=1.35 ==> 1.10246214138 2.01157181875


#    + week_day
#{'time': '10_11_2018_20_19_36',
# 'submission_file': '10_11_2018_20_19_36.csv',
# 'transformed_train_df_shape': (62313, 60007),
# 'transformed_test_df_shape': (34645, 60007),
# 'features': ['author_tfidf', 'weekday_cat', 'content_tfidf'],
# 'clf': 'ridge',
# 'valid_mae': 1.1048569125492349,                       / ~~~~~~~~~~!!!
# 'np.expm1_valid_mae': 2.0187924865803941}

#    + month
#{'time': '10_11_2018_20_28_08',
# 'submission_file': '10_11_2018_20_28_08.csv',
# 'transformed_train_df_shape': (62313, 60012),
# 'transformed_test_df_shape': (34645, 60012),
# 'features': ['author_tfidf', 'month_cat', 'content_tfidf'],
# 'clf': 'ridge',
# 'valid_mae': 1.1053952767209936,                     / ~~~~~~~~~~~~~~!!!
# 'np.expm1_valid_mae': 2.0204181338530027}

#    + hour
# {'time': '10_11_2018_20_38_18',
# 'submission_file': '10_11_2018_20_38_18.csv',
# 'transformed_train_df_shape': (62313, 60024),
# 'transformed_test_df_shape': (34645, 60024),
# 'features': ['author_tfidf', 'hour_val', 'content_tfidf'],
# 'clf': 'ridge',
# 'valid_mae': 1.1055528149574172,                    / ~~~~~~~~~~~~~~!!!
# 'np.expm1_valid_mae': 2.0208940026818558}


#     + month q1-4
#{'time': '10_11_2018_20_49_47',
# 'submission_file': '10_11_2018_20_49_47.csv',
# 'transformed_train_df_shape': (62313, 60004),
# 'transformed_test_df_shape': (34645, 60004),
# 'features': ['author_tfidf',
#  'mon_q1',
#  'mon_q2',
#  'mon_q3',
#  'mon_q4',
#  'content_tfidf'],
# 'clf': 'ridge',
# 'valid_mae': 1.1056139385100618,                    / ~~~~~~~~~~~~~~!!!
# 'np.expm1_valid_mae': 2.0210786560987417}

#      + tag (10000)
#{'time': '10_11_2018_21_09_50',
# 'submission_file': '10_11_2018_21_09_50.csv',
# 'transformed_train_df_shape': (62313, 70000),
# 'transformed_test_df_shape': (34645, 70000),
# 'features': ['author_tfidf', 'tags_tfidf', 'content_tfidf'],
# 'clf': 'ridge',
# 'valid_mae': 1.1080918138654721,          / ~~~~~~~~~~~~~~!!!
# 'np.expm1_valid_mae': 2.0285737946220319}

#      + tag (1,2) (10000)
#  {'time': '10_11_2018_21_14_12',
# 'submission_file': '10_11_2018_21_14_12.csv',
# 'transformed_train_df_shape': (62313, 70000),
# 'transformed_test_df_shape': (34645, 70000),
# 'features': ['author_tfidf', 'tags_tfidf', 'content_tfidf'],
# 'clf': 'ridge',
# 'valid_mae': 1.1164527950655041,
# 'np.expm1_valid_mae': 2.0540017965742479}

#      + tag (2,3) (10000)           1.12841240232 2.09074574222
#      + tag (1,3) (10000)           1.11755385777 2.05736629597
#      + tag (3,4) (10000)           1.11304357351 2.04360775567


#      + domain(10000)
#{'time': '10_11_2018_21_46_16',
# 'submission_file': '10_11_2018_21_46_16.csv',
# 'transformed_train_df_shape': (62313, 60292),
# 'transformed_test_df_shape': (34645, 60292),
# 'features': ['author_tfidf', 'domain_tfidf', 'content_tfidf'],
# 'clf': 'ridge',
# 'valid_mae': 1.1020590853409493,
# 'np.expm1_valid_mae': 2.010358231121804}  /!!!!!!!!!!!!!!

#     + domain + week_day
#{'time': '10_11_2018_21_58_03',
# 'submission_file': '10_11_2018_21_58_03.csv',
# 'transformed_train_df_shape': (62313, 60299),
# 'transformed_test_df_shape': (34645, 60299),
# 'features': ['author_tfidf', 'domain_tfidf', 'weekday_cat', 'content_tfidf'],
# 'clf': 'ridge',
# 'valid_mae': 1.1021357893686812,
# 'np.expm1_valid_mae': 2.0105891465790067}


# --------------------------------------------------------------------------




#{'time': '09_11_2018_22_30_10',
# 'submission_file': '09_11_2018_22_30_10.csv',
# 'transformed_train_df_shape': (62313, 60000),
# 'transformed_test_df_shape': (34645, 60000),
# 'features': ['title_tfidf', 'content_tfidf'],
# 'clf': 'ridge',
# 'valid_mae': 1.2156653138903533,
# 'np.expm1_valid_mae': 2.3725371138604436}
# ==> 1.57830
# --------------------------------------------------------------------------
# {'submission_file': 'med_baseline_1.csv',
# 'transformed_train_df_shape': (62313, 50000),
# 'transformed_test_df_shape': (34645, 50000),
# 'clf': 'ridge',
# 'valid_mae': 1.1945453278258193,
# 'np.expm1_valid_mae': 2.3020560761228199}
# ==> 1.80926 corr => 1.53729

# --------------------------------------------------------------------------
def med_baselin1_1():
    experiment_name = 'med_baseline_1'
    experiment = {}
    experiment['submission_file'] = experiment_name + '.csv'
    experiment['features'] = [v[0] for v in transform_pipeline.steps[0][1].transformer_list]

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

#   ==> 1.53729
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

# --------------------------------------------------------------------------
#{'time': '09_11_2018_22_30_10',
# 'submission_file': '09_11_2018_22_30_10.csv',
# 'transformed_train_df_shape': (62313, 60000),
# 'transformed_test_df_shape': (34645, 60000),
# 'features': ['title_tfidf', 'content_tfidf'],
# 'clf': 'ridge',
# 'valid_mae': 1.2156653138903533,
# 'np.expm1_valid_mae': 2.3725371138604436}
# ==> 1.57830

# --------------------------------------------------------------------------
def sub2():
    experiment_name = time.strftime("%d_%m_%Y_%H_%M_%S")
    experiment = {}
    experiment['time'] = experiment_name
    experiment['submission_file'] = experiment['time'] + '.csv'
    experiment['features'] = [v[0] for v in transform_pipeline.steps[0][1].transformer_list]

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
    X_valid = X_train_new[train_part_size:, :]
    y_valid = y_train_new[train_part_size:]



    ridge = Ridge(random_state = 17)
    ridge.fit(X_train_part, np.log1p(y_train_part))
    ridge_pred = np.expm1(ridge.predict(X_valid))


    plt.hist(y_valid, bins=30, alpha=.5, color='red',
             label='true', range=(0, 10))
    plt.hist(ridge_pred, bins=30, alpha=.5, color='green',
             label='pred', range=(0, 10))
    plt.legend()


    valid_mae = mean_absolute_error(y_valid, ridge_pred)
    print(valid_mae, np.expm1(valid_mae))
    experiment['clf'] = 'ridge'
    experiment['valid_mae'] = valid_mae
    experiment['np.expm1_valid_mae'] = np.expm1(valid_mae)
    experiments[experiment['time']] = experiment

    with open('medium_experiments.pickle', 'wb') as f:
        pickle.dump(experiments, f)

    ridge.fit(X_train_new, np.log1p(y_train_new))
    ridge_test_pred = np.expm1(ridge.predict(X_test_new))

    # ==> predict
    ridge_test_pred_corrected = \
        ridge_test_pred + (all_zero_mae - ridge_test_pred.mean())
    print(ridge_test_pred_corrected.mean(), all_zero_mae)

    write_submission_file(prediction=ridge_test_pred_corrected,
                          filename=experiment['submission_file'])
    # <== predict


# --------------------------------------------------------------------------
# {'time': '09_11_2018_22_35_49',
# 'submission_file': '09_11_2018_22_35_49.csv',
# 'transformed_train_df_shape': (62313, 90000),
# 'transformed_test_df_shape': (34645, 90000),
# 'features': ['author_tfidf', 'tags_tfidf', 'title_tfidf', 'content_tfidf'],
# 'clf': 'ridge',
# 'valid_mae': 1.1639669716812813,
# 'np.expm1_valid_mae': 2.2026127840842649}
#   ==> 1.54272
# --------------------------------------------------------------------------
def sub3():
    experiment_name = time.strftime("%d_%m_%Y_%H_%M_%S")
    experiment = {}
    experiment['time'] = experiment_name
    experiment['submission_file'] = experiment['time'] + '.csv'

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
    X_valid = X_train_new[train_part_size:, :]
    y_valid = y_train_new[train_part_size:]



    ridge = Ridge(random_state = 17)
    ridge.fit(X_train_part, np.log1p(y_train_part))
    ridge_pred = np.expm1(ridge.predict(X_valid))


    plt.hist(y_valid, bins=30, alpha=.5, color='red',
             label='true', range=(0, 10))
    plt.hist(ridge_pred, bins=30, alpha=.5, color='green',
             label='pred', range=(0, 10))
    plt.legend()


    valid_mae = mean_absolute_error(y_valid, ridge_pred)
    print(valid_mae, np.expm1(valid_mae))
    experiment['clf'] = 'ridge'
    experiment['valid_mae'] = valid_mae
    experiment['np.expm1_valid_mae'] = np.expm1(valid_mae)
    experiments[experiment['time']] = experiment

    with open('medium_experiments.pickle', 'wb') as f:
        pickle.dump(experiments, f)

    ridge.fit(X_train_new, np.log1p(y_train_new))
    ridge_test_pred = np.expm1(ridge.predict(X_test_new))

    # ==> predict
    ridge_test_pred_corrected = \
        ridge_test_pred + (all_zero_mae - ridge_test_pred.mean())
    print(ridge_test_pred_corrected.mean(), all_zero_mae)

    write_submission_file(prediction=ridge_test_pred_corrected,
                          filename=experiment['submission_file'])
    # <== predict


# --------------------------------------------------------------------------
#    content (50000) + author_tfidf( (2,2) +10000)
#    {'time': '10_11_2018_18_24_58',
# 'submission_file': '10_11_2018_18_24_58.csv',
# 'transformed_train_df_shape': (62313, 60000),
# 'transformed_test_df_shape': (34645, 60000),
# 'features': ['author_tfidf', 'content_tfidf'],
# 'clf': 'ridge',
# 'valid_mae': 1.1047306784298387,
# 'np.expm1_valid_mae': 2.0184114360205081}
#    ==> 1.50395
# --------------------------------------------------------------------------
def sub_1_50():
    experiment_name = time.strftime("%d_%m_%Y_%H_%M_%S")
    experiment = {}
    experiment['time'] = experiment_name
    experiment['submission_file'] = experiment['time'] + '.csv'

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
    X_valid = X_train_new[train_part_size:, :]
    y_valid = y_train_new[train_part_size:]



    ridge = Ridge(random_state = 17)
    ridge.fit(X_train_part, np.log1p(y_train_part))
    ridge_pred = np.expm1(ridge.predict(X_valid))
    #ridge.fit(X_train_part, y_train_part)
    #ridge_pred = ridge.predict(X_valid)



    plt.hist(y_valid, bins=30, alpha=.5, color='red',
             label='true', range=(0, 10))
    plt.hist(ridge_pred, bins=30, alpha=.5, color='green',
             label='pred', range=(0, 10))
    plt.legend()
    plt.show()


    valid_mae = mean_absolute_error(y_valid, ridge_pred)
    print(valid_mae, np.expm1(valid_mae))
    experiment['clf'] = 'ridge'
    experiment['valid_mae'] = valid_mae
    experiment['np.expm1_valid_mae'] = np.expm1(valid_mae)
    experiments[experiment['time']] = experiment

    with open('medium_experiments.pickle', 'wb') as f:
        pickle.dump(experiments, f)

    ridge.fit(X_train_new, np.log1p(y_train_new))
    #ridge.fit(X_train_new, y_train_new)
    ridge_test_pred = np.expm1(ridge.predict(X_test_new))
    #ridge_test_pred = ridge.predict(X_test_new)

    # ==> predict
    ridge_test_pred_corrected = \
        ridge_test_pred + (all_zero_mae - ridge_test_pred.mean())
    print(ridge_test_pred_corrected.mean(), all_zero_mae)

    write_submission_file(prediction=ridge_test_pred_corrected,
                          filename=experiment['submission_file'])
    # <== predict


# -------------------------------------------------------------------------
#{'time': '10_11_2018_22_07_11',
# 'submission_file': '10_11_2018_22_07_11.csv',
# 'transformed_train_df_shape': (62313, 60292),
# 'transformed_test_df_shape': (34645, 60292),
# 'features': ['author_tfidf', 'domain_tfidf', 'content_tfidf'],
# 'clf': 'ridge',
# 'valid_mae': 1.0994555735512541,
# 'np.expm1_valid_mae': 2.0025309216434302}
#    ==> 1.49401
# -------------------------------------------------------------------------
def sub_1_494():
    experiment_name = time.strftime("%d_%m_%Y_%H_%M_%S")
    experiment = {}
    experiment['time'] = experiment_name
    experiment['submission_file'] = experiment['time'] + '.csv'

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
    X_valid = X_train_new[train_part_size:, :]
    y_valid = y_train_new[train_part_size:]



    ridge = Ridge(random_state = 17, alpha=1.35)
    ridge.fit(X_train_part, np.log1p(y_train_part))
    ridge_pred = np.expm1(ridge.predict(X_valid))


    #ridge_params = {
    #        'alpha': np.arange(0.1, 3, step=0.05) }

    #clf = Ridge(random_state = 17)
    #clf_grid_searcher = RidgeCV(alphas=ridge_params['alpha'],
    #        estimator=clf,
    #        param_grid=ridge_params,
    #        scoring='neg_mean_absolute_error',
    #        n_jobs=-1,
    #        cv=5
    #        verbose=10
    #        )

    #clf_grid_searcher.fit(X_train_part, np.log1p(y_train_part))
    #clf_grid_searcher.alpha_
    #print(clf_grid_searcher.score(X_train_new, y_train_new))
    #print(clf_grid_searcher.best_score_, clf_grid_searcher.best_params_)
    #ridge_pred = np.expm1(clf_grid_searcher.predict(X_valid))

    plt.hist(y_valid, bins=30, alpha=.5, color='red',
             label='true', range=(0, 10))
    plt.hist(ridge_pred, bins=30, alpha=.5, color='green',
             label='pred', range=(0, 10))
    plt.legend()
    plt.show()


    valid_mae = mean_absolute_error(y_valid, ridge_pred)
    print(valid_mae, np.expm1(valid_mae))
    experiment['clf'] = 'ridge'
    experiment['valid_mae'] = valid_mae
    experiment['np.expm1_valid_mae'] = np.expm1(valid_mae)
    experiments[experiment['time']] = experiment

    with open('medium_experiments.pickle', 'wb') as f:
        pickle.dump(experiments, f)



    #ridge = Ridge(random_state=17, **clf_grid_searcher.best_params_)
    #ridge.fit(X_train_part, np.log1p(y_train_part))
    #ridge_pred = np.expm1(ridge.predict(X_valid))


    ridge.fit(X_train_new, np.log1p(y_train_new))
    #ridge.fit(X_train_new, y_train_new)
    ridge_test_pred = np.expm1(ridge.predict(X_test_new))
    #ridge_test_pred = ridge.predict(X_test_new)

    # ==> predict
    ridge_test_pred_corrected = \
        ridge_test_pred + (all_zero_mae - ridge_test_pred.mean())
    print(ridge_test_pred_corrected.mean(), all_zero_mae)

    write_submission_file(prediction=ridge_test_pred_corrected,
                          filename=experiment['submission_file'])
    # <== predict


# --------------------------------------------------------------------------
#    con_auth+dom
# 'transformed_train_df_shape': (62313, 60292),
# 'transformed_test_df_shape': (34645, 60292),
# 'features': ['author_tfidf', 'domain_tfidf', 'content_tfidf']}
# Ridge valid mae: 1.0994555735512541
# LGM valid mae: 1.1647642350341432
# Mix valid mae: 1.0954004894293936
#     ==> 1.46802
# --------------------------------------------------------------------------
def sub_mix_1():
    experiment_name = time.strftime("%d_%m_%Y_%H_%M_%S")
    experiment = {}
    experiment['time'] = experiment_name
    experiment['submission_file'] = experiment['time'] + '.csv'

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
    X_valid = X_train_new[train_part_size:, :]
    y_valid = y_train_new[train_part_size:]


    ridge = Ridge(random_state = 17, alpha=1.35)
    ridge.fit(X_train_part, np.log1p(y_train_part))
    ridge_pred = np.expm1(ridge.predict(X_valid))


    lgb_x_train_part = lgb.Dataset(
            X_train_part.astype(np.float32),
            label=np.log1p(y_train_part))

    lgb_x_valid = lgb.Dataset(
            X_valid.astype(np.float32),
            label=np.log1p(y_valid))

    param = {'num_leaves': 31, 'num_trees': 100, 'objective': 'mean_absolute_error',
            'metric': 'mae'}

    num_round = 100
    bst_lgb = lgb.train(param,
                        lgb_x_train_part,
                        num_round,
                        valid_sets=[lgb_x_valid],
                        early_stopping_rounds=20)

    lgb_pred = np.expm1(
            bst_lgb.predict(
                    X_valid.astype(np.float32),
                    num_iteration=bst_lgb.best_iteration))

    plt.hist(y_valid, bins=30, alpha=.5, color='red', label='true', range=(0,10));
    plt.hist(ridge_pred, bins=30, alpha=.5, color='green', label='Ridge', range=(0,10));
    plt.hist(lgb_pred, bins=30, alpha=.5, color='blue', label='Lgbm', range=(0,10));
    plt.legend();
    ridge_valid_mae = mean_absolute_error(y_valid, ridge_pred)
    print('Ridge valid mae: {}'.format(ridge_valid_mae))
    lgb_valid_mae = mean_absolute_error(y_valid, lgb_pred)
    print('LGM valid mae: {}'.format(lgb_valid_mae))
    print('Mix valid mae: {}'.format(
            mean_absolute_error(y_valid, .6 * lgb_pred + .4 * ridge_pred)))

    ridge.fit(X_train_new, np.log1p(y_train_new));
    lgb_x_train = lgb.Dataset(X_train_new.astype(np.float32),
                              label=np.log1p(y_train_new))
    num_round = 50
    bst_lgb = lgb.train(param, lgb_x_train, num_round)

    ridge_test_pred = np.expm1(ridge.predict(X_test_new))
    lgb_test_pred = np.expm1(bst_lgb.predict(X_test_new.astype(np.float32)))

    mix_test_pred = .6 * lgb_test_pred + .4 * ridge_test_pred
    # ==> predict
    mix_test_pred_corrected = \
        mix_test_pred + (all_zero_mae - mix_test_pred.mean())
    print(mix_test_pred_corrected.mean(), all_zero_mae)
    write_submission_file(prediction=mix_test_pred_corrected,
                          filename=experiment['submission_file'])
    # <== predict


# --------------------------------------------------------------------------

def train_lgm(Xtrain, ytrain, Xvalid, yvalid, Xtest):
    experiment_name = 'train_lgm' + time.strftime("%d_%m_%Y_%H_%M_%S")
    experiment = {}
    experiment['time'] = experiment_name

    print(Xtrain.shape)

    experiment['transformed_train_df_shape'] = Xtrain.shape
    experiment['features'] = [v[0] for v in transform_pipeline.steps[0][1].transformer_list]

    lgb_x_train_part = lgb.Dataset(
            Xtrain.astype(np.float32),
            label=np.log1p(ytrain))

    lgb_x_valid = lgb.Dataset(
            Xvalid.astype(np.float32),
            label=np.log1p(yvalid))

    param = {'num_leaves': 31,
             'num_trees': 500,
             'objective': 'mean_absolute_error',
             'metric': 'mae'}

    num_round = 100
    bst_lgb = lgb.train(param,
                        lgb_x_train_part,
                        num_round,
                        valid_sets = [lgb_x_valid],
                        early_stopping_rounds=20)

    lgb_pred = np.expm1(
            bst_lgb.predict(
                    Xvalid.astype(np.float32),
                    num_iteration = bst_lgb.best_iteration))

    lgb_valid_mae = mean_absolute_error(yvalid, lgb_pred)
    print('LGM valid mae: {}'.format(lgb_valid_mae))

    experiment['clf'] = 'lgm'
    experiment['valid_mae'] = lgb_valid_mae
    experiment['np.expm1_valid_mae'] = np.expm1(lgb_valid_mae)

    lgb_test_pred = np.expm1(bst_lgb.predict(Xtest.astype(np.float32)))

    return bst_lgb, lgb_pred, experiment, lgb_test_pred


# --------------------------------------------------------------------------
def train_clf(clf, Xtrain, ytrain, Xvalid, yvalid, Xtest, clf_name):
    experiment_name = clf_name + time.strftime("%d_%m_%Y_%H_%M_%S")
    experiment = {}
    experiment['time'] = experiment_name

    print(Xtrain.shape)

    experiment['transformed_train_df_shape'] = Xtrain.shape
    experiment['features'] = [v[0] for v in transform_pipeline.steps[0][1].transformer_list]

    clf.fit(Xtrain, np.log1p(ytrain))
    clf_pred = np.expm1(clf.predict(Xvalid))

    valid_mae = mean_absolute_error(yvalid, clf_pred)
    print('{} valid mae: {}'.format(clf_name, valid_mae))

    experiment['clf'] = clf_name
    experiment['valid_mae'] = valid_mae
    experiment['np.expm1_valid_mae'] = np.expm1(valid_mae)

    clf_test_pred = np.expm1(clf.predict(Xtest))

    return clf, clf_pred, experiment, clf_test_pred

# --------------------------------------------------------------------------
def train_ridge(Xtrain, ytrain, Xvalid, yvalid, Xtest):
    clf = Ridge(random_state = 17, alpha=1.35)
    return train_clf(clf, Xtrain, ytrain, Xvalid, yvalid, Xtest, 'train_ridge')

# --------------------------------------------------------------------------
def train_sgd(Xtrain, ytrain, Xvalid, yvalid, Xtest):
#    {'alpha': 1e-06,
#     'loss': 'epsilon_insensitive',
#     'max_iter': 1000,
#     'penalty': 'l2'}
    clf = SGDRegressor(random_state = 17,
                       alpha=1e-06,
                       loss='epsilon_insensitive',
                       max_iter=1000,
                       penalty='l2',
                       verbose=1
                       )

    return train_clf(clf, Xtrain, ytrain, Xvalid, yvalid, Xtest, 'train_sgd')


# --------------------------------------------------------------------------
#  LGM valid mae: 1.1470621492830722
#  Ridge valid mae: 1.0994555735512541
#  SGD valid mae: 1.1078919832261511
#  Mix valid mae: 1.0754468052550874
#ridge_experiment
#Out[181]:
#{'time': 'train_ridge11_11_2018_17_12_45',
# 'transformed_train_df_shape': (43619, 60292),
# 'features': ['author_tfidf', 'domain_tfidf', 'content_tfidf'],
# 'clf': 'train_ridge',
# 'valid_mae': 1.0994555735512541,
# 'np.expm1_valid_mae': 2.0025309216434302}
#
#lgm_experiment
#Out[182]:
#{'time': 'train_lgm11_11_2018_17_14_08',
# 'transformed_train_df_shape': (43619, 60292),
# 'features': ['author_tfidf', 'domain_tfidf', 'content_tfidf'],
# 'clf': 'lgm',
# 'valid_mae': 1.1470621492830722,
# 'np.expm1_valid_mae': 2.148928226177985}
#
#sgd_experiment
#Out[183]:
#{'time': 'train_sgd11_11_2018_17_25_31',
# 'transformed_train_df_shape': (43619, 60292),
# 'features': ['author_tfidf', 'domain_tfidf', 'content_tfidf'],
# 'clf': 'train_sgd',
# 'valid_mae': 1.1078919832261511,
# 'np.expm1_valid_mae': 2.0279686532493293}
#    ==> 1.46854
# --------------------------------------------------------------------------
def sub_rg_lg_sgd_1():
    experiment_name = time.strftime("%d_%m_%Y_%H_%M_%S")
    experiment = {}
    experiment['time'] = experiment_name
    experiment['submission_file'] = experiment['time'] + '.csv'

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
    X_valid = X_train_new[train_part_size:, :]
    y_valid = y_train_new[train_part_size:]

    # train ridge
    ridge, ridge_pred, ridge_experiment, ridge_test_pred = train_ridge(
            X_train_part,
            y_train_part,
            X_valid,
            y_valid,
            X_test_new)
    len(ridge_test_pred)
    lgm, lgb_pred, lgm_experiment, lgm_test_pred = train_lgm(
            X_train_part,
            y_train_part,
            X_valid,
            y_valid,
            X_test_new)
    len(lgm_test_pred)
    sgd, sgd_pred, sgd_experiment, sgd_test_pred = train_sgd(
            X_train_part,
            y_train_part,
            X_valid,
            y_valid,
            X_test_new)
    len(sgd_test_pred)
    mix_pred = .4 * lgb_pred + .3 * ridge_pred + 0.3 * sgd_pred
    mix_test_pred = .4 * lgm_test_pred + .3 * ridge_test_pred + 0.3 * sgd_test_pred
    len(mix_test_pred)

    print('LGM valid mae: {}'.format(lgm_experiment['valid_mae']))
    print('Ridge valid mae: {}'.format(ridge_experiment['valid_mae']))
    print('SGD valid mae: {}'.format(sgd_experiment['valid_mae']))
    print('Mix valid mae: {}'.format(mean_absolute_error(y_valid, mix_pred)))

    plt.hist(y_valid, bins=30, alpha=.5, color='red', label='true', range=(0,10));
    plt.hist(ridge_pred, bins=30, alpha=.5, color='green', label='Ridge', range=(0,10));
    plt.hist(sgd_pred, bins=30, alpha=.5, color='blue', label='SGD', range=(0,10));
    plt.hist(mix_pred, bins=30, alpha=.5, color='maroon', label='mixed', range=(0,10));
    plt.legend();


    experiments[lgm_experiment['time']] = lgm_experiment
    experiments[ridge_experiment['time']] = ridge_experiment
    experiments[sgd_experiment['time']] = sgd_experiment

    with open('medium_experiments.pickle', 'wb') as f:
        pickle.dump(experiments, f)

    # ==> predict
    test_pred_corrected = \
        mix_test_pred + (all_zero_mae - mix_pred.mean())
    write_submission_file(prediction=test_pred_corrected,
                          filename=experiment['submission_file'])
    # <== predict

























ridge = Ridge(random_state = 17, alpha=1.35)
ridge.fit(X_train_part, np.log1p(y_train_part))
ridge_pred = np.expm1(ridge.predict(X_valid))


lgb_x_train_part = lgb.Dataset(
        X_train_part.astype(np.float32),
        label=np.log1p(y_train_part))

lgb_x_valid = lgb.Dataset(
        X_valid.astype(np.float32),
        label=np.log1p(y_valid))

param = {'num_leaves': 31,
         'num_trees': 500,
         'objective': 'mean_absolute_error',
         'metric': 'mae'}

num_round = 100
bst_lgb = lgb.train(param,
                    lgb_x_train_part,
                    num_round,
                    valid_sets=[lgb_x_valid],
                    early_stopping_rounds=20)

lgb_pred = np.expm1(
        bst_lgb.predict(
                X_valid.astype(np.float32),
                num_iteration=bst_lgb.best_iteration))
lgb_valid_mae = mean_absolute_error(y_valid, lgb_pred)
print('LGM valid mae: {}'.format(lgb_valid_mae))

plt.hist(y_valid, bins=30, alpha=.5, color='red', label='true', range=(0,10));
plt.hist(ridge_pred, bins=30, alpha=.5, color='green', label='Ridge', range=(0,10));
plt.hist(lgb_pred, bins=30, alpha=.5, color='blue', label='Lgbm', range=(0,10));
plt.legend();
ridge_valid_mae = mean_absolute_error(y_valid, ridge_pred)
print('Ridge valid mae: {}'.format(ridge_valid_mae))

print('Mix valid mae: {}'.format(
        mean_absolute_error(y_valid, .6 * lgb_pred + .4 * ridge_pred)))

ridge.fit(X_train_new, np.log1p(y_train_new));
lgb_x_train = lgb.Dataset(X_train_new.astype(np.float32),
                          label=np.log1p(y_train_new))
num_round = 50
bst_lgb = lgb.train(param, lgb_x_train, num_round)

ridge_test_pred = np.expm1(ridge.predict(X_test_new))
lgb_test_pred = np.expm1(bst_lgb.predict(X_test_new.astype(np.float32)))

mix_test_pred = .6 * lgb_test_pred + .4 * ridge_test_pred
# ==> predict
mix_test_pred_corrected = \
    mix_test_pred + (all_zero_mae - mix_test_pred.mean())
print(mix_test_pred_corrected.mean(), all_zero_mae)
write_submission_file(prediction=mix_test_pred_corrected,
                      filename=experiment['submission_file'])
# <== predict


sgd_params = {
    'alpha': (0.00001, 0.000001),
    'penalty': ('l2', 'elasticnet'),
    'max_iter': (50, 100, 500, 1000),
    'loss': ['epsilon_insensitive', 'huber']
}
sgd = SGDRegressor(random_state = 17,
                     loss='epsilon_insensitive',
                     max_iter=1000,
                     verbose=1)
sgd = GridSearchCV(
        sgd,
        sgd_params,
        cv=5,
        n_jobs=-1,
        verbose=10)

#sgd = RandomForestRegressor(
#        random_state = 17,
#        n_estimators=200,
#        criterion='mae',
#        verbose=10)
sgd.fit(X_train_part, np.log1p(y_train_part))
sgd_sgrid = sgd
sgd_sgrid_pred = np.expm1(sgd_sgrid.predict(X_valid))
print(sgd_sgrid.score(X_train_new, y_train_new))
valid_mae = mean_absolute_error(y_valid, sgd_sgrid_pred)
print(valid_mae, np.expm1(valid_mae))

sgd = SGDRegressor(random_state = 17,
                     verbose=1,
                     **sgd_sgrid.best_params_)
sgd.fit(X_train_part, np.log1p(y_train_part))
sgd_pred = np.expm1(sgd.predict(X_valid))
print(sgd.score(X_train_new, y_train_new))
valid_mae = mean_absolute_error(y_valid, sgd_pred)
print(valid_mae, np.expm1(valid_mae))

#mix_pred = .6 * lgb_pred + .4 * ridge_pred# + 0.3 * sgd_sgrid_pred
mix_pred = .4 * lgb_pred + .3 * ridge_pred + 0.3 * sgd_sgrid_pred
print('Mix valid mae: {}'.format(mean_absolute_error(y_valid, mix_pred)))

plt.hist(y_valid, bins=30, alpha=.5, color='red', label='true', range=(0,10));
plt.hist(ridge_pred, bins=30, alpha=.5, color='green', label='Ridge', range=(0,10));
plt.hist(sgd_pred, bins=30, alpha=.5, color='blue', label='SGD', range=(0,10));
plt.hist(mix_pred, bins=30, alpha=.5, color='maroon', label='mixed', range=(0,10));
plt.legend();

#ridge_params = {
#        'alpha': np.arange(0.1, 3, step=0.05) }

#clf = Ridge(random_state = 17)
#clf_grid_searcher = RidgeCV(alphas=ridge_params['alpha'],
#        estimator=clf,
#        param_grid=ridge_params,
#        scoring='neg_mean_absolute_error',
#        n_jobs=-1,
#        cv=5
#        verbose=10
#        )

#clf_grid_searcher.fit(X_train_part, np.log1p(y_train_part))
#clf_grid_searcher.alpha_
#print(clf_grid_searcher.score(X_train_new, y_train_new))
#print(clf_grid_searcher.best_score_, clf_grid_searcher.best_params_)
#ridge_pred = np.expm1(clf_grid_searcher.predict(X_valid))

plt.hist(y_valid, bins=30, alpha=.5, color='red',
         label='true', range=(0, 10))
plt.hist(ridge_pred, bins=30, alpha=.5, color='green',
         label='pred', range=(0, 10))
plt.legend()
plt.show()


valid_mae = mean_absolute_error(y_valid, ridge_pred)
print(valid_mae, np.expm1(valid_mae))
experiment['clf'] = 'ridge'
experiment['valid_mae'] = valid_mae
experiment['np.expm1_valid_mae'] = np.expm1(valid_mae)
experiments[experiment['time']] = experiment

with open('medium_experiments.pickle', 'wb') as f:
    pickle.dump(experiments, f)



#ridge = Ridge(random_state=17, **clf_grid_searcher.best_params_)
#ridge.fit(X_train_part, np.log1p(y_train_part))
#ridge_pred = np.expm1(ridge.predict(X_valid))


ridge.fit(X_train_new, np.log1p(y_train_new))
#ridge.fit(X_train_new, y_train_new)
ridge_test_pred = np.expm1(ridge.predict(X_test_new))
#ridge_test_pred = ridge.predict(X_test_new)

# ==> predict
ridge_test_pred_corrected = \
    ridge_test_pred + (all_zero_mae - ridge_test_pred.mean())
print(ridge_test_pred_corrected.mean(), all_zero_mae)

write_submission_file(prediction=ridge_test_pred_corrected,
                      filename=experiment['submission_file'])
# <== predict




















import collections
from keras.layers.core import Dense, Dropout, SpatialDropout1D
from keras.layers.convolutional import Conv1D
from keras.layers.embeddings import Embedding
from keras.layers.pooling import GlobalMaxPooling1D
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import nltk


NUM_EPOCHS = 5
VOCAB_SIZE = 5000
EMBED_SIZE = 100
NUM_FILTERS = 256
NUM_WORDS = 3
BATCH_SIZE = 64
HIDDEN_SIZE = 10


def preprocess_nn():
    experiment_name = 'med_lstm'
    experiment = {}
    experiment['submission_file'] = experiment_name + '.csv'
    experiment['features'] = 'content'


    counter = collections.Counter()
    maxlen = 0
    def tokenizer(x):
        global maxlen
        words = [x.lower() for x in nltk.word_tokenize(x)]
        if len(words) > maxlen:
            maxlen = len(words)
        for word in words:
            counter[word] += 1

    # word indexes
    def xs(data):
        xs = []

        def word_wids(x):
            words = [x.lower() for x in nltk.word_tokenize(x)]
            wids = [word2index[word] for word in words]
            xs.append(wids)

        data.apply(word_wids)
        return xs

    # tokenize it!
    train_df['content'].apply(tokenizer)
    test_df['content'].apply(tokenizer)

    # word indexes
    word2index = collections.defaultdict(int)
    for wid, word in enumerate(counter.most_common(VOCAB_SIZE)):
        word2index[word[0]] = wid + 1

    # vocalubary
    vocab_size = len(word2index) + 1
    index2word = { v:k for k, v in word2index.items() }


    xs_content_train = xs(train_df['content'])
    xs_content_test = xs(test_df['content'])

    # paddings
    XX = pad_sequences(xs_content_train, maxlen=maxlen)
    XX_test = pad_sequences(xs_content_test, maxlen=maxlen)


#with open('medium_experiments_nn_preprocessor_data_0.pickle', 'rb') as f:
#    maxlen = pickle.load(f)

# load variables from file
#with open('medium_experiments_nn_preprocessor_data1.pickle', 'rb') as f:
#    index2word = pickle.load(f)
#    word2index = pickle.load(f)
    #with open('medium_experiments_nn_preprocessor_data3.pickle', 'rb') as f:
#    xs_content_train = pickle.load(f)
#with open('medium_experiments_nn_preprocessor_data4.pickle', 'rb') as f:
#    xs_content_test = pickle.load(f)
#XX = np.load('medium_experiments_nn_preprocessor_data2.npz')
#aa = XX[XX.files[0]]
#XX.close()
#XX = aa
#XX_test = np.load('medium_experiments_nn_preprocessor_data2_1.npz')
#bb = XX_test[XX_test.files[0]]
#XX_test.close()
#XX_test = bb


# save variables to file
#with open('medium_experiments_nn_preprocessor_data_0.pickle', 'wb') as f:
#    pickle.dump(maxlen, f)
#with open('medium_experiments_nn_preprocessor_data1.pickle', 'wb') as f:
#    pickle.dump(index2word, f)
#    pickle.dump(word2index, f)
#np.savez_compressed('medium_experiments_nn_preprocessor_data2', XX)
#np.savez_compressed('medium_experiments_nn_preprocessor_data2_1', XX_test)
#with open('medium_experiments_nn_preprocessor_data3.pickle', 'wb') as f:
#    pickle.dump(xs_content_train, f)
#with open('medium_experiments_nn_preprocessor_data4.pickle', 'wb') as f:
#    pickle.dump(xs_content_test, f)

#del xs_content_train, xs_content_test
#gc.collect()

yy = train_df['target']


# train + test
XX_train, XX_valid, yy_train, yy_valid = train_test_split(XX,
                                                          yy,
                                                          test_size=0.3,
                                                          random_state=17)


# create NN model
model = Sequential()
model.add(Embedding(vocab_size, EMBED_SIZE, input_length=maxlen))
#model.add(Conv1D(filters=NUM_FILTERS, kernel_size=NUM_WORDS, activation='relu'))
#model.add(GlobalMaxPooling1D())
model.add(LSTM(HIDDEN_SIZE, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1))

model.compile(loss='mean_absolute_error',
              optimizer='adam',
              metrics=['mae'])

history = model.fit(XX_train, np.log1p(yy_train),
                    batch_size=BATCH_SIZE,
                    epochs=NUM_EPOCHS,
                    validation_data=(XX_valid, np.log1p(yy_valid)))

score = model.evaluate(XX_valid, np.log1p(yy_valid), verbose=1)
print('Test score: {}'.format(score[0]))
print('Test mae: {}'.format(score[1]))


plt.subplot(211)
plt.title('mae')
plt.plot(history.history['mae'], color='g', label='Train')
plt.plot(history.history['val_mae'], color='b', label='Test')
plt.legeng(loc='best')
plt.subplot(212)
plt.title('LOSS')
plt.plot(history.history['loss'], color='g', label='Train')
plt.plot(history.history['val_mae'], color='b', label='Test')
plt.legeng(loc='best')
plt.tight_layout()
plt.show()

model_pred = np.expm1(model.predict(XX_valid))
valid_mae = mean_absolute_error(yy_valid, model_pred)
print(valid_mae, np.expm1(valid_mae))

experiment['clf'] = 'LSTM'
experiment['valid_mae'] = valid_mae
experiment['np.expm1_valid_mae'] = np.expm1(valid_mae)
experiments[time.strftime("%d_%m_%Y_%H:%M:%S")] = experiment

with open('medium_experiments.pickle', 'wb') as f:
    pickle.dump(experiments, f)


# train using full data
model.fit(XX, yy)
model_test_pred = model.predict(XX_test)


# ==> predict
model_test_pred_corrected = \
    model_test_pred + (all_zero_mae - model_test_pred.mean())
print(model_test_pred_corrected.mean(), all_zero_mae)

write_submission_file(prediction=model_test_pred_corrected,
                      filename=experiment['submission_file'])
# <== predict





















