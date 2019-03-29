# %%
"""
This file converts the full RegulonDB text file into the CSV files used by PredCRP.pl.
It only accepts 42-length sequences with strong evidence of activation XOR repression by CRP,
just like the actual paper.
"""
import csv
import pandas as pd
import numpy as np

# %%
columns = []
data = []
with open("BindingSiteSet_RegulonDB_v10.5.txt") as fd:
    rd = csv.reader(fd, delimiter="\t", quotechar='"')
    start_columns = False
    start_data = False
    for row in rd:
        if not start_data:
            if len(row) > 1:
                start_data = True
            elif start_columns:
                columns.append(row[0])
            elif row[0] == '# Columns:':
                start_columns = True
        if start_data:
            data.append(row)
    print(columns)
    print(data)

# %%
df = pd.DataFrame(data, columns=columns)

# %%
df = df[df['# (2) TF name'] == 'CRP']
df = df[df['# (14) Evidence confidence level (Confirmed, Strong, Weak)'] == 'Strong']
df = df[df['# (12) TF-bs sequence (upper case)'].str.len() == 42]
not_dual = df['# (9) Gene expression effect caused by the TF bound to the  TF-bs (+ activation, - repression, +- dual, ? unknown)'] != '+-'
not_unknown = df['# (9) Gene expression effect caused by the TF bound to the  TF-bs (+ activation, - repression, +- dual, ? unknown)'] != '?'
df = df[not_dual & not_unknown]
df = df[['# (12) TF-bs sequence (upper case)', '# (11) Center position of TF-bs, relative to Transcription Start Site', '# (8) Transcription unit regulated by the TF',
         '# (9) Gene expression effect caused by the TF bound to the  TF-bs (+ activation, - repression, +- dual, ? unknown)']]
df.replace('', np.nan, inplace=True)
df.dropna(inplace=True)
# %%
df.columns = ['CRPBS', 'Distance of Center Position of CRPBS to TSS',
              'Transcription Unit', 'Regulatory Role']
df.reset_index(drop=True, inplace=True)

# %%
df.to_csv('CRPBS_v10.5.csv', index=False)
