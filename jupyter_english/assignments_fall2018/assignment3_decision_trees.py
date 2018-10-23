
# coding: utf-8

# <center>
# <img src="../../img/ods_stickers.jpg" />
#
# ## [mlcourse.ai](mlcourse.ai) – Open Machine Learning Course
# Author: [Yury Kashnitskiy](https://yorko.github.io) (@yorko). Edited by Anna Tarelina (@feuerengel). This material is subject to the terms and conditions of the [Creative Commons CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license. Free use is permitted for any non-commercial purpose.

# # <center>Assignment #3. Fall 2018
# ## <center> Decision trees for classification and regression

# **In this assignment, we will find out how a decision tree works in a
# regression task, then will build and tune classification decision trees
# for identifying heart diseases.
# Fill in the missing code in the cells marked "You code here" and answer
# the questions in the [web form](https://docs.google.com/forms/d/1hsrNFSiRsvgB27gMbXfQWpq8yzNhLZxuh_VSzRz7XhI).**



import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz


# ## 1. A simple example of regression using decision trees

# Let's consider the following one-dimensional regression problem. It is
# needed to build the function $a(x)$ to approximate original dependency
# $y = f(x)$ using mean-squared error $min \sum_i {(a(x_i) - f(x_i))}^2$.


X = np.linspace(-2, 2, 7)
y = X ** 3

plt.scatter(X, y)
plt.xlabel(r'$x$')
plt.ylabel(r'$y$');


# Let's make several steps to build the decision tree. Let's choose the
# symmetric thresholds equal to 0, 1.5 and -1.5 for partitioning. In the
# case of a regression task, the leaf outputs mean answer for all observations
# in this leaf.

# Let's start from tree of depth 0 that contains all train observations. How
# will predictions of this tree look like for $x \in [-2, 2]$? Create the
# appropriate plot using a pen, paper and Python if it is needed (without using `sklearn`).




# Let's split the data according to the following condition $[x < 0]$.
# It gives us the tree of depth 1 with two leaves. Let's create a similar
# plot for predictions of this tree.



# You code here
def predict_tree(X, y, x):
    # t = [0, -1.5, 1.5]
    if x < 0:
        if x < -1.5:
            return y[X < -1.5].mean()
        else:
            return y[(X < 0) & (X >= -1.5)].mean()
    else:
        if x < 1.5:
            return y[(X >= 0) & (X < 1.5)].mean()
        else:
            return y[X > 1.5].mean()


# In the decision tree algorithm, the feature and the threshold for splitting
# are chosen according to some criterion. The commonly used criterion for
# regression is based on variance: $$\large Q(X, y, j, t) = D(X, y) - \dfrac{|X_l|}{|X|} D(X_l, y_l) - \dfrac{|X_r|}{|X|} D(X_r, y_r),$$
# where $\large X$ and $\large y$ are a feature matrix and a target vector
# (correspondingly) for training instances in a current node, $\large X_l, y_l$ and $\large X_r, y_r$ are splits of samples $\large X, y$ into two parts w.r.t. $\large [x_j < t]$ (by $\large j$-th feature and threshold $\large t$), $\large |X|$, $\large |X_l|$, $\large |X_r|$ (or, the same, $\large |y|$, $\large |y_l|$, $\large |y_r|$) are sizes of appropriate samples, and $\large D(X, y)$ is variance of answers $\large y$ for all instances in $\large X$:
# $$\large D(X) = \dfrac{1}{|X|} \sum_{j=1}^{|X|}(y_j – \dfrac{1}{|X|}\sum_{i = 1}^{|X|}y_i)^2$$
# Here $\large y_i = y(x_i)$ is the answer for the $\large x_i$ instance.
# Feature index $\large j$ and threshold $\large t$ are chosen to maximize
# the value of criterion  $\large Q(X, y, j, t)$ for each split.
#
# In our 1D case,  there's only one feature so $\large Q$ depends only on
# threshold $\large t$ and training data $\large X$ and $\large y$.
# Let's designate it $\large Q_{1d}(X, y, t)$ meaning that the criterion
# no longer depends on feature index $\large j$, i.e. in 1D case $\large j = 0$.
#
# Create the plot of criterion $\large Q_{1d}(X, y, t)$  as a function of
# threshold value $t$ on the interval $[-1.9, 1.9]$.

def regression_var_criterion(X, y, t):
    nX = y.size
    nXl = y[X < t].size
    nXr = nX - nXl
    return y.var() - nXl / nX * y[X < t].var() - nXr / nX * y[X >= t].var()


xx = np.linspace(-1.9, 1.9, 100)
yy = [regression_var_criterion(X, y, t) for t in xx]

fig, ax = plt.subplots(figsize=(8,4))
ax.plot(xx, yy)
plt.xlabel('t')
plt.ylabel('Var Criterion')




# **<font color='red'>Question 1.</font> Is the threshold value $t = 0$ optimal
# according to the variance criterion?**
# - Yes --
# - No

# Then let's make splitting in each of the leaves' nodes. In the left branch
# (where previous split was $x < 0$) using the criterion $[x < -1.5]$, in the
# right branch (where previous split was $x \geqslant 0$) with the following
# criterion $[x < 1.5]$. It gives us the tree of depth 2 with 7 nodes and 4 leaves.
# Create the plot of these tree predictions for $x \in [-2, 2]$.


xx = np.linspace(-2, 2, 100)
yy = [predict_tree(X, y, t) for t in xx]

fig, ax = plt.subplots(figsize=(8,4))
ax.plot(xx, yy)
plt.xlabel('x')
plt.ylabel('Prediction')
plt.scatter(X, y)



# **<font color='red'>Question 2.</font> How many segments are there on the
# plot of tree predictions in the interval [-2, 2] (it is necessary to count
#                                          only horizontal lines)?**
# - 2
# - 3 --
# - 4
# - 5

# ## 2. Building a decision tree for predicting heart diseases
# Let's read the data on heart diseases. The dataset can be downloaded from
 #the course repo from [here](https://github.com/Yorko/mlcourse.ai/blob/master/data/mlbootcamp5_train.csv) by clicking on `Download` and then selecting `Save As` option.
#
# **Problem**
#
# Predict presence or absence of cardiovascular disease (CVD) using the
 #patient examination results.
#
# **Data description**
#
# There are 3 types of input features:
#
# - *Objective*: factual information;
# - *Examination*: results of medical examination;
# - *Subjective*: information given by the patient.
#
# | Feature | Variable Type | Variable      | Value Type |
# |---------|--------------|---------------|------------|
# | Age | Objective Feature | age | int (days) |
# | Height | Objective Feature | height | int (cm) |
# | Weight | Objective Feature | weight | float (kg) |
# | Gender | Objective Feature | gender | categorical code |
# | Systolic blood pressure | Examination Feature | ap_hi | int |
# | Diastolic blood pressure | Examination Feature | ap_lo | int |
# | Cholesterol | Examination Feature | cholesterol | 1: normal, 2: above normal, 3: well above normal |
# | Glucose | Examination Feature | gluc | 1: normal, 2: above normal, 3: well above normal |
# | Smoking | Subjective Feature | smoke | binary |
# | Alcohol intake | Subjective Feature | alco | binary |
# | Physical activity | Subjective Feature | active | binary |
# | Presence or absence of cardiovascular disease | Target Variable | cardio | binary |
#
# All of the dataset values were collected at the moment of medical examination.



df = pd.read_csv('../../data/mlbootcamp5_train.csv',
                 index_col='id', sep=';')

df.head().T
df.info()
#df['gender'] = df['gender'].astype('category')


# Transform the features: create "age in years" (full age) and also create
# 3 binary features based on `cholesterol` and 3 more on `gluc`, where they
# are equal to 1, 2 or 3. This method is called dummy-encoding or
# One Hot Encoding (OHE). It is more convenient to use `pandas.get_dummmies.`.
# There is no need to use the original features `cholesterol` and `gluc` after encoding.


df['fullage'] = (df['age'] / 365).astype('int')
df['gluc'].unique()
df['cholesterol'].unique()
df = pd.concat([df, pd.get_dummies(df['cholesterol'], prefix='cholesterol')], axis=1)
df = pd.concat([df, pd.get_dummies(df['gluc'], prefix='gluc')], axis=1)
df.drop(['cholesterol', 'gluc'], axis=1, inplace=True)
df.head().T

# Split data into train and holdout parts in the proportion of 7/3 using `sklearn.model_selection.train_test_split` with `random_state=17`.

X = df.drop(['cardio'], axis=1)
y = df['cardio']

X_train, X_valid, y_train, y_valid = train_test_split(X.values, y, test_size=0.3, random_state=17)

tree = DecisionTreeClassifier(max_depth=3, random_state=17)
tree.fit(X_train, y_train)


# Train the decision tree on the dataset `(X_train, y_train)` with max depth
# equals to 3 and `random_state=17`. Plot this tree with
# `sklearn.tree.export_graphviz`, `dot` and `pydot`. You don't need to use
# quotes in the file names in order to make it work in a jupyter notebook.
# The commands starting from the exclamation mark are terminal commands that
# are usually run in terminal/command line.

# use .dot format to visualize a tree
from sklearn.tree import export_graphviz
from graphviz import Source
from IPython.display import Image

pydot_graph = pydotplus.graph_from_dot_data(export_graphviz(tree, out_file=None, filled=True, feature_names=X.columns))
#pydot_graph.set_size('"15"')
#Source(export_graphviz(tree, out_file=None, filled=True, feature_names=X.columns))
Image(pydot_graph.create_png())



# **<font color='red'>Question 3.</font> What 3 features are used to make
# predictions in the created decision tree?**
# - weight, height, gluc=3
# - smoke, age, gluc=3
# - age, weight, chol=3
# - age, ap_hi, chol=3 --

# Make predictions for holdout data `(X_valid, y_valid)` with the trained
# decision tree. Calculate accuracy.


tree_pred = tree.predict(X_valid)
acc1 = accuracy_score(y_valid, tree_pred) # 0.72
acc1



# You code here


# Set up the depth of the tree using cross-validation on the dataset
# `(X_train, y_train)` in order to increase quality of the model.
# Use `GridSearchCV` with 5 folds. Fix `random_state=17` and change
#  `max_depth` from 2 to 10.


tree_params = {'max_depth': list(range(2, 11))}

tree_grid = GridSearchCV(DecisionTreeClassifier(random_state=17), tree_params, cv=5, n_jobs=-1, verbose=True)
tree_grid.fit(X_train, y_train)


# Draw the plot to show how mean accuracy is changing in regards to `max_depth`
# value on cross-validation.


from sklearn.model_selection import cross_val_score
cv_scores, holdout_scores = [], []
n_maxdepth = list(range(1, 11, 1))

for max_depth in n_maxdepth:
    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=17)

    cv_scores.append(np.mean(cross_val_score(tree, X_train, y_train, cv=5)))
    tree.fit(X_train, y_train)

    holdout_scores.append(accuracy_score(y_valid, tree.predict(X_valid)))

plt.plot(n_maxdepth, cv_scores, label='CV')
plt.plot(n_maxdepth, holdout_scores, label='Holdout')
plt.xlabel('max_depth')
plt.title('Tree accuracy')
plt.legend();



# Print the best value of `max_depth` where the mean value of cross-validation
# quality metric reaches maximum. Also compute accuracy on holdout data.
# All these computations are possible to make using the trained instance
# of the class `GridSearchCV`.

acc2 = tree_grid.best_score_
tree_grid.best_params_, tree_grid.best_score_ # ({'max_depth': 6}, 0.73195918367346935)


# **<font color='red'>Question 4.</font> Is there a local maximum of accuracy
# on the built validation curve? Did `GridSearchCV` help to tune `max_depth`
# so that there's been at least 1% change in holdout accuracy?**
# (check out the expression (acc2 - acc1) / acc1 * 100%, where acc1 and acc2
# are accuracies on holdout data before and after tuning `max_depth` with
# `GridSearchCV` respectively)?
# - yes, yes --
# - yes, no
# - no, yes
# - no, no

(acc2 - acc1) / acc1 * 100 # 1.48


# Take a look at the SCORE table to estimate ten-year risk of fatal
# cardiovascular disease in Europe.
# [Source paper](https://academic.oup.com/eurheartj/article/24/11/987/427645).
#
# <img src='../../img/SCORE2007-eng.png' width=70%>
#
# Create binary features according to this picture:
# - $age \in [40,50), \ldots age \in [60,65) $ (4 features)
# - systolic blood pressure: $ap\_hi \in [120,140), ap\_hi \in [140,160), ap\_hi \in [160,180),$ (3 features)
#
# If the values of age or blood pressure don't fall into any of the intervals
# then all binary features will be equal to zero. Then we create decision tree
# with these features and additional ``smoke``, ``cholesterol``  and ``gender``
# features. Transform the ``cholesterol`` to 3 binary features according to it's
# 3 unique values ( ``cholesterol``=1,  ``cholesterol``=2 and  ``cholesterol``=3).
# This method is called dummy-encoding or One Hot Encoding (OHE). Transform
# the ``gender`` from 1 and 2 into 0 and 1. It is better to rename it to
# ``male`` (0 – woman, 1 – man). In general, this is typically done with
# ``sklearn.preprocessing.LabelEncoder`` but here in case of only 2 unique
# values it's not necessary.
#
# Finally the decision tree is built using 12 binary features
# (without original features).
#
# Create a decision tree with the limitation `max_depth=3` and train it on
# the whole train data. Use the `DecisionTreeClassifier` class with fixed
# `random_state=17`, but all other arguments (except for `max_depth` and
# `random_state`) should be set by default.
#
# **<font color='red'>Question 5.</font> What binary feature is the most
# important for heart disease detection (it is placed in the root of the tree)?**
# - Systolic blood pressure from 160 to 180 (mmHg)
# - Gender male / female
# - Systolic blood pressure from 140 to 160 (mmHg) --
# - Age from 50 to 55 (years)
# - Smokes / doesn't smoke
# - Age from 60 to 65 (years)



def divider(x, low, high):
    if low <= x < high:
        return 1
    else:
        return 0

#df['age40_50'] = np.where((40 <= df['fullage']) & (df['fullage'] < 50), 'yes', 'no')
age_columns = ['age40_50', 'age50_55', 'age55_60', 'age60_65']
age_bounds = [ [40, 50], [50, 55], [55, 60], [60, 65] ]

for col_name, bound in zip(age_columns, age_bounds):
    df[col_name] = df['fullage'].apply(lambda age: divider(age, bound[0], bound[1]))

sys_columns = ['sys120_140', 'sys140_160', 'sys160_180']
sys_bounds = [ [120, 140], [140, 160], [160, 180] ]

for col_name, bound in zip(sys_columns, sys_bounds):
    df[col_name] = df['ap_hi'].apply(lambda sysbp: divider(sysbp, bound[0], bound[1]))

#df = pd.concat([df, pd.get_dummies(df['cholesterol'], prefix='cholesterol')], axis=1)
df['gender'] = df['gender'].map({1: 0, 2: 1})
df.rename(columns={'gender': 'male'}, inplace=True)

generated_df = df[['male', 'smoke', 'cholesterol_1', 'cholesterol_2', 'cholesterol_3']]
generated_df = pd.concat([generated_df, df[age_columns]], axis=1)
generated_df = pd.concat([generated_df, df[sys_columns]], axis=1)

X_gen_train = generated_df
y_gen_train = df['cardio']
tree = DecisionTreeClassifier(max_depth=3, random_state=17)
tree.fit(X_gen_train, y_gen_train)
#tree.feature_importances_

pydot_graph = pydotplus.graph_from_dot_data(export_graphviz(tree, out_file=None, filled=True, feature_names=X_gen_train.columns))
Image(pydot_graph.create_png())


