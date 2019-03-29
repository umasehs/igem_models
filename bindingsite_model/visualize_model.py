# %%
"""
A bunch of different tools that helped me visualize the model after it had been trained.
"""
from joblib import load
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from xgboost import to_graphviz
from yellowbrick.classifier import classification_report
from yellowbrick.classifier import ROCAUC
from yellowbrick.classifier import ClassPredictionError
from yellowbrick.classifier import DiscriminationThreshold
import shap

shap.initjs()
# %%
clf = load('clf.joblib')

# %%
df = pd.read_csv('CRPBS_v10.5.csv')

# %%
class_names = ['Other', 'Binding Site']

# %%
n_rows = len(df)
n_bases = 22
bs = np.empty((n_rows, n_bases * 4))
o = np.empty((n_rows * (n_bases - 1), n_bases * 4))
label_encoder = LabelEncoder()
onehot_encoder = OneHotEncoder(sparse=False)

sequences = df['CRPBS']

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
bs = bs[~np.all(bs == 0, axis=1)]
o = o[~np.all(o == 0, axis=1)]

binding_sites = bs
other = o
binding_sites_labels = np.ones(binding_sites.shape[0], dtype=np.uint8)
other_labels = np.zeros(other.shape[0], dtype=np.uint8)
X = np.concatenate((binding_sites, other))
y = np.concatenate((binding_sites_labels, other_labels))

# %%
to_graphviz(clf, num_trees=0, rankdir='LR')

# %%
classification_report(clf, X, y)

# %%
visualizer = ROCAUC(clf, classes=class_names)
visualizer.score(X, y)
visualizer.poof()

# %%
visualizer = ClassPredictionError(clf, classes=class_names)
visualizer.score(X, y)
visualizer.poof()

# %%
visualizer = DiscriminationThreshold(clf)
visualizer.fit(X, y)
visualizer.poof()

# %%
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X)

# %%
shap.force_plot(explainer.expected_value, shap_values, X)

# %%
shap.force_plot(explainer.expected_value, shap_values[0, :], X[0, :])

# %%
shap.summary_plot(shap_values, X, plot_type="bar")

# %%
shap.summary_plot(shap_values, X)

# %%
top_inds = np.argsort(-np.sum(np.abs(shap_values), 0))
for i in range(10):
    shap.dependence_plot(top_inds[i], shap_values, X)
