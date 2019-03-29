# %%
"""
In this file, I just play around with the features to try and extract the best ones.
This is only helpful if you want to see why I chose the features I did.
"""
from joblib import load
import numpy as np
import sklearn.model_selection
import sklearn.metrics
import sklearn.ensemble
from xgboost import XGBClassifier
from yellowbrick.features import Rank1D
from yellowbrick.target import FeatureCorrelation
from yellowbrick.features import RFECV
import shap

# %%
shap.initjs()

# %%
with open('all_feature_list.txt') as file_object:
    lines = [line[:-1] for line in file_object.readlines()]
    verbose_feature_names = lines
feature_names = np.arange(len(verbose_feature_names))

# %%
class_names = ['+', '-']
verbose_class_names = ['Activator', 'Repressor']

# %%
X = load('X.joblib')
y = load('y.joblib')


# %%
visualizer = Rank1D()
visualizer.fit(X, y)
visualizer.transform(X)
visualizer.poof()

# %%
feat = visualizer.features_[visualizer.ranks_ > 0.5]
print(feat)
X = X[feat]

# %%
seed = 15
test_size = 0.25
cv = 5
Xt, Xv, yt, yv = \
    sklearn.model_selection.train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed)

# %%
clf = XGBClassifier(
    max_depth=7,
    n_estimators=1000,
    scale_pos_weight=50000,
    subsample=0.5,
    colsample_bylevel=0.5,
    colsample_bytree=0.5,
    seed=seed,
)
scores = sklearn.model_selection.cross_validate(
    clf, Xt, yt, return_estimator=True, cv=cv)
clf = scores['estimator'][np.argmax(scores['test_score'])]
print(np.max(scores['test_score']))

# %%
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(Xv)

# %%
shap.summary_plot(shap_values, Xv, plot_type="bar")


# %%
feat = feature_names[feat][np.mean(abs(shap_values), axis=0) > 0.45]
print(feat)
X = X[feat]


# %%
test_size = 0.33
cv = 3
Xt, Xv, yt, yv = \
    sklearn.model_selection.train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed)

# %%
clf = XGBClassifier(
    max_depth=7,
    n_estimators=1000,
    scale_pos_weight=50000,
    subsample=0.5,
    colsample_bylevel=0.5,
    colsample_bytree=0.5,
    seed=seed,
)
scores = sklearn.model_selection.cross_validate(
    clf, Xt, yt, return_estimator=True, cv=cv)
clf = scores['estimator'][np.argmax(scores['test_score'])]
print(np.max(scores['test_score']))

# %%
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(Xv)

# %%
shap.summary_plot(shap_values, Xv, plot_type="bar")


# %%
feat = feature_names[feat][np.mean(abs(shap_values), axis=0) > 1]
print(feat)
X = X[feat]


# %%
test_size = 0.33
cv = 3
Xt, Xv, yt, yv = \
    sklearn.model_selection.train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed)
viz = RFECV(XGBClassifier(
    max_depth=7,
    n_estimators=1000,
    scale_pos_weight=50000,
    subsample=0.5,
    colsample_bylevel=0.5,
    colsample_bytree=0.5,
    seed=seed,
), cv=cv)
viz.fit(Xv, yv)
viz.poof()


# %%
feat = feat[viz.ranking_ < 5]
print(feat)
X = X[feat]


# %%
test_size = 0.33
cv = 3
Xt, Xv, yt, yv = \
    sklearn.model_selection.train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed)
viz = RFECV(XGBClassifier(
    max_depth=7,
    n_estimators=1000,
    scale_pos_weight=50000,
    subsample=0.5,
    colsample_bylevel=0.5,
    colsample_bytree=0.5,
    seed=seed,
), cv=cv)
viz.fit(Xt, yt)
viz.poof()

# %%
feat = feat[viz.support_]
print(feat)
X = X[feat]

# %%
test_size = 0.33
cv = 3
Xt, Xv, yt, yv = \
    sklearn.model_selection.train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed)
viz = RFECV(XGBClassifier(
    max_depth=7,
    n_estimators=1000,
    scale_pos_weight=50000,
    subsample=0.5,
    colsample_bylevel=0.5,
    colsample_bytree=0.5,
    seed=seed,
), cv=cv)
viz.fit(Xt, yt)
viz.poof()

# %%
feat = feat[viz.ranking_ < 3]
print(feat)
X = X[feat]


# %%
seed = 7
test_size = 0.25
cv = 3
Xt, Xv, yt, yv = \
    sklearn.model_selection.train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed)

# %%
clf = XGBClassifier(
    max_depth=3,
    n_estimators=1000,
    scale_pos_weight=5000,
    subsample=0.85,
    colsample_bylevel=0.85,
    colsample_bytree=0.85,
    seed=seed,
)
scores = sklearn.model_selection.cross_validate(
    clf, Xt, yt, return_estimator=True, cv=cv)
clf = scores['estimator'][np.argmax(scores['test_score'])]
print(np.max(scores['test_score']))

# %%
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(Xv)

# %%
shap.summary_plot(shap_values, Xv, plot_type="bar")

# %%
feat = feature_names[feat][np.mean(abs(shap_values), axis=0) > 0.55]
print(feat)
X = X[feat]


# %%
visualizer = FeatureCorrelation(method='mutual_info-classification')
visualizer.fit(X, y)
visualizer.poof()

# %%
# This step doesn't always produce the same result, idk why.
feat = visualizer.features_[visualizer.scores_ > 0.04]
X = X[feat]

# %%
# Our final 10 features:
# [263, 268, 287, 288, 300, 302, 307, 308, 313, 315]
print(feat)
