# %%
"""
A bunch of different tools that helped me visualize the model after it had been trained.
"""
from joblib import load
import numpy as np
import sklearn.metrics
import sklearn.ensemble
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
keep = [263, 268, 287, 288, 300, 302, 307, 308, 313, 315]

# %%
seed = 15
test_size = 0.33
Xt, Xv, yt, yv = \
    sklearn.model_selection.train_test_split(
        X[keep], y, test_size=test_size, stratify=y, random_state=seed)

# %%
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(Xv)

# %%
shap.force_plot(explainer.expected_value, shap_values, Xv)

# %%
shap.force_plot(explainer.expected_value, shap_values[0, :], Xv.iloc[0, :])

# %%
shap.summary_plot(shap_values, Xv, plot_type="bar")

# %%
shap.summary_plot(shap_values, Xv)

# %%
top_inds = np.argsort(-np.sum(np.abs(shap_values), 0))
for i in range(10):
    shap.dependence_plot(top_inds[i], shap_values, Xv)
