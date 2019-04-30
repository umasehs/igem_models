# %%
"""
Quickly tests the model that classifies whether CRP will activate or repress its target.

To use this model on some arbitrary sequences,
just replace the RegulonDB text file in this folder with an identically-formatted file
(containing your preferred sequences), then run this script.
"""

import subprocess
import joblib
import pandas as pd
import numpy as np
import sklearn.model_selection
import sklearn.metrics

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
# Use the Perl script to generate the 392 features of CRP binding sites, using the Regulon DB text file as input.
# The features are written to "BindingSiteSet_RegulonDB_vXY.Z_CRP_strong_svmformat.txt", where XY.Z is the database version number.
cmd = "cd test/PredCRP; perl feature_extraction_BindingSites.pl -input ../BindingSiteSet_RegulonDB_v10.5.txt -TF CRP -evidence 1 -length 42; cd .."
subprocess.call(cmd, shell=True)

# %%
# Extract the 392 features from the svmformat.txt file.
X, y = parse_datafile(
    "BindingSiteSet_RegulonDB_v10.5_CRP_strong_svmformat.txt")

X = pd.DataFrame(X)

# %%
# These are the 0-based indices of our 10 selected features.
keep = [263, 268, 287, 288, 300, 302, 307, 308, 313, 315]
X = X[keep]

# %%
# Split the data into training and validation sets.
seed = 15
test_size = 0.33
Xt, Xv, yt, yv = \
    sklearn.model_selection.train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed)


# %%
# Load the pretrained classifier
clf = joblib.load('clf.joblib')

# %%
# Use the model to try to classify binding sites as activated or repressed by CRP
y_hat = clf.predict(Xt)
print("Training Accuracy:", sklearn.metrics.accuracy_score(yt, y_hat))
y_hat = clf.predict(Xv)
print("Validation Accuracy:", sklearn.metrics.accuracy_score(yv, y_hat))
print("Confusion Matrix (all data): \n",
      sklearn.metrics.confusion_matrix(y, clf.predict(X)))


# My results:
# Training Accuracy: 0.952
# Validation Accuracy: 0.9365079365079365
# Confusion Matrix (all data):
# [[136  9]
# [ 1 42]]
