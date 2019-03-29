# %%
"""
A bunch of different tools that helped me visualize the data before I started working on the model.
"""
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import pandas as pd
from yellowbrick.features import ParallelCoordinates
from yellowbrick.features import Rank2D
from yellowbrick.target import ClassBalance
from yellowbrick.target import FeatureCorrelation
from yellowbrick.features import RadViz
from yellowbrick.features.pca import PCADecomposition
from yellowbrick.features.importances import FeatureImportances
from yellowbrick.features import Rank1D

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
visualizer = ClassBalance(labels=class_names)
visualizer.fit(y)
visualizer.poof()

# %%
visualizer = ParallelCoordinates()
visualizer.fit_transform(X, y)
visualizer.poof()

# %%
visualizer = Rank1D()
visualizer.fit(X, y)
visualizer.transform(X)
visualizer.poof()

# %%
visualizer = Rank2D()
visualizer.fit_transform(X)
visualizer.poof()

# %%
visualizer = FeatureCorrelation()
visualizer.fit(X, y)
visualizer.poof()

# %%
visualizer = FeatureCorrelation(method='mutual_info-classification')
visualizer.fit(X, y)
visualizer.poof()

# %%
visualizer = RadViz(classes=class_names)
visualizer.fit(X, y)
visualizer.transform(X)
visualizer.poof()

# %%
colors = np.array(['r' if yi else 'b' for yi in y])
visualizer = PCADecomposition(color=colors, proj_features=True)
visualizer.fit_transform(X, y)
visualizer.poof()
visualizer = PCADecomposition(
    scale=True, color=colors, proj_dim=3, proj_features=True)
visualizer.fit_transform(X, y)
visualizer.poof()

# %%
viz = FeatureImportances(GradientBoostingClassifier(), relative=False)
viz.fit(X, y)
viz.poof()
