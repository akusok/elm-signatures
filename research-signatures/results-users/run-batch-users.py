#    for j in $(seq 1 99); do echo $j; python run-experiment-batch.py > res${j}.txt; done

from hpelm import HPELM
import os
import pickle
import random
import numpy as np
import pandas as pd


def load_batch(j, loo=False):
    """Load batch `j` of `users` list, as Numpy array.
    
    loo (bool, default=False): whether to return single LOO file per user separately.
    """    
    data = X0[X0.uid.isin(users[j]) & (X0.sig_true == 1)]
    
    if loo:
        # get LOO split
        t = data.loc[:, 'uid':'fid']
        t = t.sample(frac=1.0, replace=False)
        fid_loo = t.groupby('uid').first()['fid']

        data_loo = data[data.fid.isin(fid_loo)]
        data = data[~data.fid.isin(fid_loo)]
        Y_loo = data_loo.loc[:, 'uid']
        Y_loo = U0.loc[Y_loo.index].to_numpy()
        Z_loo = data_loo.loc[:, '0':'1023'].to_numpy()

    Y = data.loc[:, 'uid']
    Y = U0.loc[Y.index].to_numpy()
    Z = data.loc[:, '0':'1023'].to_numpy()
    
    zm = Z.mean()
    zs = Z.std().clip(min=0.5)
    Z = (Z - zm) / zs
    Z = Z.clip(min=-5, max=5)

    if loo:
        Z_loo = (Z_loo - zm) / zs
        Z_loo = Z_loo.clip(min=-5, max=5)
        return Z, Y, Z_loo, Y_loo
        
    return Z, Y



if __name__ == "__main__":
    data_folder = "/Users/akusok/wrkdir/research-signatures-data/MCYTDB"
    overlap = '50p'
    samples = 10000000 // 30

    data_file = "/Users/akusok/wrkdir/research-signatures-data/MCYTD_overlap{}.pkl".format(overlap)

    with open(data_file, 'rb') as f:
        X0 = pickle.load(f)

    X0 = X0.reset_index(drop=True)
    U0 = pd.get_dummies(X0.uid)

    uu = list(X0.uid.unique())
    random.shuffle(uu)
    users = [uu[:50]]
    users.extend([uu[i:i+5] for i in range(50,75,5)])
    
    
    model = HPELM(1024, 75, norm=10**0.8, batch=10000)
    model.add_neurons(10000-1024, 'tanh')
    model.add_neurons(1024, 'lin')
    Xp, Yp = load_batch(0, loo=False)
    model.add_data(Xp, Yp)
    
    res = []
    for k in range(1,6):
        print(k)
        x, y, xv, yv = load_batch(k, loo=True)

        model.add_data(x, y)
        model.nnet.solve()

        yh1 = model.predict(xv).argmax(1)
        yv1 = yv.argmax(1)
        b = pd.DataFrame(np.vstack((yv1, yh1)).T).groupby(0)
        val_topk = b.apply(lambda a: list(a[1].value_counts().index)[:10])
        val_true = b.count().index

        val_data = {}
        for topk, yy in zip(val_topk, val_true):
            val_data[yy] = [int(yy in topk[:i]) for i in (1,3,5,10)]

        res_k = pd.DataFrame.from_dict(val_data, orient='index', columns=['top1', 'top3', 'top5' ,'top10'])

        res.append(res_k)


    R = pd.concat(res, axis=0)
    print(R)
    print(R.mean(axis=0))
    print("Done!")