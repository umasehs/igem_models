"""
The PredCRP tool outputs its predictions in CSV files.
This script reads those files and outputs the accuracy of PredCRP's predictions.
"""
# %%
import pandas as pd
import numpy as np

# %%
predictresult = pd.read_csv("CRPBS_v10.5_PredictResult.csv")
roles = predictresult[['Regulatory Role', 'Pred role']]
roles.replace(to_replace='activation', value='+', inplace=True)
roles.replace(to_replace='repression', value='-', inplace=True)

print(roles)

# %%
correct = roles['Regulatory Role'] == roles['Pred role']
accuracy = (np.sum(correct)/len(correct))
print(accuracy)
