#    for j in $(seq 1 99); do echo $j; python run-experiment-batch.py > res${j}.txt; done

from hpelm import HPELM
import os
import pickle
import random
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score


def load_batch(j, loo=False):
    """Load batch `j` of `users` list, as Numpy array.
    
    loo (bool, default=False): whether to return single LOO file per user separately.
    """    
    data = X0[X0.uid.isin(users[j])]
    
    if loo:
        # get LOO split
        t = data.loc[:, 'uid':'fid']
        t = t.sample(frac=1.0, replace=False)
        fid_loo = t.groupby('uid').first()['fid']

        data_loo = data[data.fid.isin(fid_loo)]
        data = data[~data.fid.isin(fid_loo)]
        Y_loo = data_loo.loc[:, 'sig_true'].to_numpy()
        U_loo = data_loo.loc[:, 'uid']
        Z_loo = data_loo.loc[:, '0':'1023'].to_numpy()

    Y = data.loc[:, 'sig_true'].to_numpy()
    Z = data.loc[:, '0':'1023'].to_numpy()
    
    zm = Z.mean()
    zs = Z.std().clip(min=0.5)
    Z = (Z - zm) / zs
    Z = Z.clip(min=-5, max=5)

    if loo:
        Z_loo = (Z_loo - zm) / zs
        Z_loo = Z_loo.clip(min=-5, max=5)
        return Z, Y, Z_loo, Y_loo, U_loo
        
    return Z, Y



if __name__ == "__main__":
    data_folder = "/Users/akusok/wrkdir/research-signatures-data/MCYTDB"
    overlap = '50p'
    samples = 10000000 // 30

    data_file = "/Users/akusok/wrkdir/research-signatures-data/MCYTD_overlap{}.pkl".format(overlap)

    with open(data_file, 'rb') as f:
        X0 = pickle.load(f)

    X0 = X0.reset_index(drop=True)

    uu = list(X0.uid.unique())
    random.shuffle(uu)
    users = [uu[:50]]
    users.extend([uu[i:i+5] for i in range(50,75,5)])
    
    
    model = HPELM(1024, 1, norm=10**0.8, batch=10000)
    model.add_neurons(10000-1024, 'tanh')
    model.add_neurons(1024, 'lin')
    Xp, Yp = load_batch(0, loo=False)
    model.add_data(Xp, Yp)
    model.nnet.solve()
    
    res = []
    for k in range(1,6):
        print(k)
        x, y, xv, yv, uv = load_batch(k, loo=True)

        yh1 = model.predict(xv)
        yv1 = yv[:,None]
        res_k = pd.DataFrame(np.hstack((yv1, yh1)), index=uv).groupby('uid').mean()
        res.append(res_k)

        model.add_data(x, y)
        model.nnet.solve()


    R = pd.concat(res, axis=0)
    pt = R[0]
    ph = R[1]

    print(R)
    print('accuracy {:.3f}'.format(accuracy_score(pt>0.5, ph>0.5)))
    print('auc {:.3f}'.format(roc_auc_score(pt, ph)))
    print("Done!")