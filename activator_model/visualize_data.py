# %%
"""
A bunch of different tools that helped me visualize the data before I started working on the model.
"""
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
from sklearn.ensemble import GradientBoostingClassifier

# %%
with open('all_feature_list.txt') as file_object:
    lines = [line[:-2] for line in file_object.readlines()]
    verbose_feature_names = lines
feature_names = np.arange(len(verbose_feature_names))

# %%
class_names = ['+', '-']
verbose_class_names = ['Activator', 'Repressor']

# %%


def parse_datafile(filepath):
    with open(filepath, 'r') as file_object:
        X, y = map(list, zip(*[parse_line(line)
                               for line in file_object.readlines()]))
    return np.array(X), np.array(y)


def parse_line(line):
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
X, y = parse_datafile(
    "BindingSiteSet_RegulonDB_v10.5_CRP_strong_svmformat.txt")

# %%
X = pd.DataFrame(X, columns=feature_names)
y = np.ravel(pd.DataFrame(y, columns=['+ vs -']))

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
