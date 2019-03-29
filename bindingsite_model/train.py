# %%
"""
Train a model to find the CRP binding site within the promoter sequence.

It only accepts 42-length sequences with strong evidence of activation XOR repression by CRP,
and it predicts the 22-length binding sites for CRP,
because these were the lengths used in the PredCRP paper.

Using half of our data for training and half for validation,
the trained model achieves 96.8% training accuracy, and 90.4% validation accuracy.
"""
from xgboost import XGBClassifier
import sklearn.ensemble
import sklearn.metrics
import sklearn.model_selection
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
from joblib import dump


# %%
def evaluate(clf, eval_df, out_string):
    """
    Given the classifier and the DataFrame containing the genetic sequences,
    evaluates the classifier's ability to correctly find the binding site,
    then prints the classifier's accuracy (prefixed by the out_string).
    """
    sequences = eval_df['CRPBS']
    n_rows = len(eval_df)

    # In our dataset, the binding site is always preceded by 10 bases
    y = np.full((n_rows), 10)
    y_hat = np.empty((n_rows))

    # For each sequence in the DataFrame
    for index, sequence in sequences.iteritems():
        b = sequence
        n = n_bases  # n_bases = 22

        # Split the sequence into every possible n-length string (these are called n-grams)
        ngrams = [b[i:i+n] for i in range(len(b)-n+1)]
        n_ngrams = len(ngrams)
        proba = np.empty((n_ngrams))

        for i in range(n_ngrams):
            # One-hot encode each base in each n-gram
            ngram = ngrams[i]
            seq_array = np.array(list(ngram.lower()))
            integer_encoded_seq = label_encoder.transform(seq_array)
            integer_encoded_seq = integer_encoded_seq.reshape(
                len(integer_encoded_seq), 1)
            onehot_encoded_seq = onehot_encoder.transform(integer_encoded_seq)
            features = onehot_encoded_seq.flatten()[np.newaxis, :]

            # Estimate the probability that this 22-gram is the binding site
            proba[i] = clf.predict_proba(features)[0, 1]
        # The 22-gram with the highest probability is the predicted binding site
        y_hat[index] = np.argmax(proba)
    print(out_string, sklearn.metrics.accuracy_score(y, y_hat))


# %%
df = pd.read_csv('CRPBS_v10.5.csv')

# %%
seed = 12
test_size = 0.5
cv = 10

np.random.seed(seed)

# Half our data is used for training, the other half for validation
df_train, df_validate = \
    sklearn.model_selection.train_test_split(
        df, test_size=test_size, random_state=seed)
df_train.reset_index(drop=True, inplace=True)
df_validate.reset_index(drop=True, inplace=True)


# One-hot encode the 22-grams in our training data
n_rows = len(df_train)
n_bases = 22
bs = np.empty((n_rows, n_bases * 4))  # The binding sites
o = np.empty((n_rows * (n_bases - 1), n_bases * 4))  # All the other 22-grams
label_encoder = LabelEncoder()
onehot_encoder = OneHotEncoder(sparse=False)

sequences = df_train['CRPBS']

seq_array = np.array(list('acgt'))
integer_encoded_seq = label_encoder.fit_transform(seq_array)
integer_encoded_seq = integer_encoded_seq.reshape(
    len(integer_encoded_seq), 1)
onehot_encoded_seq = onehot_encoder.fit_transform(integer_encoded_seq)

for index, sequence in sequences.iteritems():
    b = sequence
    n = n_bases
    ngrams = [b[i:i+n] for i in range(len(b)-n+1)]
    n_ngrams = len(ngrams)
    for i in range(n_ngrams):
        ngram = ngrams[i]
        seq_array = np.array(list(ngram.lower()))
        integer_encoded_seq = label_encoder.transform(seq_array)
        integer_encoded_seq = integer_encoded_seq.reshape(
            len(integer_encoded_seq), 1)
        onehot_encoded_seq = onehot_encoder.transform(integer_encoded_seq)
        features = onehot_encoded_seq.flatten()[np.newaxis, :]
        if (ngram.isupper()):
            bs[index] = features
        else:
            o[(index * n_ngrams) + i] = features

# Delete any empty rows
bs = bs[~np.all(bs == 0, axis=1)]
o = o[~np.all(o == 0, axis=1)]

binding_sites = bs
other = o
binding_sites_labels = np.ones(binding_sites.shape[0], dtype=np.uint8)
other_labels = np.zeros(other.shape[0], dtype=np.uint8)
X = np.concatenate((binding_sites, other))
y = np.concatenate((binding_sites_labels, other_labels))

# Use a gradient boosting model to try to predict which 22-grams are binding sites.
clf = XGBClassifier(
    max_depth=5,  # tree depth is max 5
    n_estimators=400,  # 400 trees
    subsample=0.5,  # subsample the training data to prevent overfitting
    colsample_bylevel=0.5,  # subsample the columns for each level of each tree
    colsample_bytree=0.5,
    random_state=seed,
)

# 10-fold validation within our training set
scores = sklearn.model_selection.cross_validate(
    clf, X, y, return_estimator=True, cv=cv)
clf = scores['estimator'][np.argmax(scores['test_score'])]

# Use the classifier on every 22-gram in a genetic sequence
# The 22-gram with the highest probability is the predicted binding site
# Print the accuracy of our predictions.
evaluate(clf, df_train, "Training Accuracy: ")
evaluate(clf, df_validate, "Validation Accuracy: ")
evaluate(clf, df, "Total Accuracy: ")

# My results:
# Training Accuracy:  0.9680851063829787
# Validation Accuracy:  0.9042553191489362
# Total Accuracy:  0.9361702127659575


# %%
# Save the model (joblib is like pickle but optimized for numpy arrays).
# It can be retrieved with joblib.load(filename).
dump(clf, 'clf.joblib')
