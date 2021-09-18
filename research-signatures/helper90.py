
import os
import numpy as np
import pandas as pd
import skelm

from sklearn.model_selection import  RandomizedSearchCV
from sklearn.utils.fixes import loguniform

import warnings
from sklearn.exceptions import DataConversionWarning, DataDimensionalityWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=DataDimensionalityWarning)

if __name__ == "__main__":
    overlap = '90p'
    samples = 200
    data_file = "/home/akusok/Documents/research-signatures/MCYTD_overlap{}_n{}.pkl".format(overlap, samples)

    x = pd.read_pickle(data_file)
    x = x.sample(100000).reset_index(drop=True)

    uid = pd.get_dummies(x.uid, prefix='user')
    sig_true = x.sig_true
    fid = x.fid

    x = np.array(x.loc[:, '0':'1023'])
    x = x - x.mean(0)
    x = x / x.std(0).clip(min=0.5)
    x = x.clip(min=-5, max=5)

    x = np.hstack((x, np.array(uid)))
    y = np.array(sig_true)
    groups = fid

    params = {
        'alpha': loguniform(1e-2, 1e+1),
        'n_neurons': loguniform(1000, 10000),
    }

    for i in range(100000):
        print(i)
        rcv = RandomizedSearchCV(skelm.ELMClassifier(), params, n_iter=100, scoring='accuracy', cv=3, refit=False)
        res = rcv.fit(x, y)
        pd.DataFrame(res.cv_results_).to_pickle("res_v3/res_{}_{}.pkl".format(overlap, i))
