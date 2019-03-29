# %%
"""
Train a model to classify whether CRP will activate or repress its target.

This script inputs the 390ish features from feature_extraction_BindingSites.pl,
but it only actually uses 10 of them.

Using 2/3 of our data for training and 1/3 for validation, and only 10 features,
the trained model achieves 95.2% training accuracy, and 93.7% validation accuracy.
Also, on the 23 extra sequences from the PredCRP paper, it achieves 100% accuracy.

While the original PredCRP model achieved 98% training and 93% validation,
this was on an older version of the database (using 12 features)
and it achieves ~87% on the newest version.
It also only predicted 22/23 test sequences accurately.
"""
from xgboost import XGBClassifier
import sklearn.ensemble
import sklearn.metrics
import sklearn.model_selection
import numpy as np
import pandas as pd
from joblib import dump

# %%
with open('all_feature_list.txt') as file_object:
    lines = [line[:-1] for line in file_object.readlines()]
    verbose_feature_names = lines
feature_names = np.arange(len(verbose_feature_names))

# %%
class_names = ['+', '-']
verbose_class_names = ['Activator', 'Repressor']

# %%


def parse_datafile(filepath):
    """
    Parses the svmformat.txt files produced by feature_extraction_BindingSites.pl.
    """
    with open(filepath, 'r') as file_object:
        X, y = map(list, zip(*[parse_line(line)
                               for line in file_object.readlines()]))
    return np.array(X), np.array(y)


def parse_line(line):
    """
    Parses each line of the svmformat.txt files, specifically the 392 features.
    """
    input_line = line.split(' ')[:-1]
    label = int(input_line[0])
    features = np.zeros((392))
    for i in input_line[1:]:
        idx, val = i.split(':', 1)
        idx = int(idx) - 1
        if idx >= 392:
            break
        features[idx] = float(val)
    return (features.tolist(), label)


# %%
# Our training data is the 392 features of CRP-binding sites from RegulonDB v10.5.
X, y = parse_datafile(
    "BindingSiteSet_RegulonDB_v10.5_CRP_strong_svmformat.txt")

X = pd.DataFrame(X, columns=feature_names)
y = np.ravel(pd.DataFrame(y, columns=['+ vs -']))


# %%
# The 23 test sequences were from RegulonDB v9.4, and were only supported by weak evidence.
Xtest, ytest = parse_datafile(
    "BindingSiteSet_RegulonDB_v9.4_CRP_weak_svmformat.txt")

Xtest = pd.DataFrame(Xtest, columns=feature_names)
ytest = np.ravel(pd.DataFrame(ytest, columns=['+ vs -']))


# %%
# I manually identified each sequence's index within the svmformat file,
# by comparing CRPBS_23weak_PredictResult.csv to 9.4_CRP_weak.tsv.
# TODO: Someone should probably double-check that I didn't make any mistakes.
# Note that array indices are 0-based, while line numbers in files are often 1-based.
twentythree = [106, 31, 32, 44, 139, 105, 135, 1, 102, 97,
               92, 87, 118, 57, 83, 75, 127, 26, 14, 128, 67, 116, 117]
# We manually set the labels for these 23,
# because the ground truth was discovered by the PredCRP researchers,
# not from RegulonDB.
y23 = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
X23 = Xtest.iloc[twentythree]

# %%
# These are the 0-based indices of our 10 selected features.
keep = [263, 268, 287, 288, 300, 302, 307, 308, 313, 315]


# %%
seed = 15
test_size = 0.33
cv = 3
Xt, Xv, yt, yv = \
    sklearn.model_selection.train_test_split(
        X[keep], y, test_size=test_size, stratify=y, random_state=seed)

# Use a gradient boosting model to try to classify binding sites as activated or repressed by CRP
clf = XGBClassifier(
    max_depth=7,  # tree depth is max 7
    n_estimators=1000,  # there are 1000 trees
    scale_pos_weight=50000,  # counteracts the class imbalance b/w activators/repressors
    subsample=0.5,  # subsample the training data to prevent overfitting
    colsample_bylevel=0.5,  # subsample the columns for each level of each tree
    colsample_bytree=0.5,
    random_state=seed,
)

# 3-fold cross-validation within our training set
scores = sklearn.model_selection.cross_validate(
    clf, Xt, yt, return_estimator=True, cv=cv)
clf = scores['estimator'][np.argmax(scores['test_score'])]

y_hat = clf.predict(Xt)
print("Training Accuracy:", sklearn.metrics.accuracy_score(yt, y_hat))
y_hat = clf.predict(Xv)
print("Validation Accuracy:", sklearn.metrics.accuracy_score(yv, y_hat))
y_hat = clf.predict(X23[keep])
print("Test Accuracy:", sklearn.metrics.accuracy_score(y23, y_hat))

# My results:
# Training Accuracy: 0.952
# Validation Accuracy: 0.9365079365079365
# Test Accuracy: 1.0

# %%
# Save the model (joblib is like pickle but optimized for numpy arrays).
# It can be retrieved with joblib.load(filename).
dump(clf, 'clf.joblib')


# %%
# Print the full names of our 10 features (names taken from all_feature_list.txt).
print(np.array(verbose_feature_names)[keep])

# %%
# Save the preprocessed data for convenience.
# dump(X, 'X.joblib')
# dump(y, 'y.joblib')
