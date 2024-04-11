#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 22:58:06 2023

@author: alain
"""

#%%

import numpy as np
import seaborn as sns
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline

import moabb
from moabb.datasets import BNCI2014009, BNCI2014001, Shin2017A
from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import MotorImagery
from moabb.datasets.utils import find_intersecting_channels

"""

http://moabb.neurotechx.com/docs/datasets.html


---------------------------------------------------------
list of motor imagery datasets

[<moabb.datasets.alex_mi.AlexMI at 0x7f18d02ad130>, 8, 16, 3 classes
 <moabb.datasets.bnci.BNCI2014001 at 0x7f18d02ade80>, 10,22, 4
 <moabb.datasets.bnci.BNCI2014002 at 0x7f18d02add90>, 15,15, 2
 <moabb.datasets.bnci.BNCI2014004 at 0x7f18d279e6d0>, 10,3, 2 classes
 <moabb.datasets.bnci.BNCI2015001 at 0x7f18d279e910>,
 <moabb.datasets.bnci.BNCI2015004 at 0x7f18d279ecd0>,
 <moabb.datasets.gigadb.Cho2017 at 0x7f18d279e9a0>,
 <moabb.datasets.Lee2019.Lee2019_MI at 0x7f18d279ec10>,
 <moabb.datasets.mpi_mi.MunichMI at 0x7f18d279eb50>,
 <moabb.datasets.upper_limb.Ofner2017 at 0x7f18d279eeb0>,
 <moabb.datasets.physionet_mi.PhysionetMI at 0x7f18d279eca0>,
 <moabb.datasets.schirrmeister2017.Schirrmeister2017 at 0x7f18d27b6100>,
 <moabb.datasets.bbci_eeg_fnirs.Shin2017A at 0x7f18d279e070>,
 <moabb.datasets.Weibo2014.Weibo2014 at 0x7f18d27b64c0>, 10, 60
 <moabb.datasets.Zhou2016.Zhou2016 at 0x7f18d27b6880>] 4, 14, 3
 
 """

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace


fmin = 8
fmax = 30
paradigm = MotorImagery(fmin=fmin, fmax=fmax)
indices = [0,1,2,3,13,14]
dsets = [paradigm.datasets[i] for i in indices]
true_labels = ['left_hand', 'right_hand', 'feet', 'tongue','rest']
#%%
select_data = 0


if select_data == 0:
    # the full dataset 
    electrodes = None
    filename ='bci_users54_5c.npz'
elif select_data == 1:
    # the dataset with only the common channels
    electrodes, dsets = find_intersecting_channels(dsets)
    electrodes = None
    filename ='bci_users40_5c_specific.npz'
elif select_data == 2:
    electrodes, dsets = find_intersecting_channels(dsets)
    filename ='bci_users40_5c_common.npz'
    
print(dsets)
user = {}
meta_info = {}
k = 0 
for i_data in range(len(dsets)):
    dset = dsets[i_data]
    if dset.code == 'Cho2017':
        dset.subject_list = dset.subject_list[:10]

    for i_sub in range(len(dset.subject_list)):
        subject = dset.subject_list[i_sub]
        if electrodes is not None:
            paradigm.channels = electrodes
        epos, labels, meta = paradigm.get_data(
                            dset,[subject], return_epochs=True)
        print(labels)
        y = np.zeros(len(labels))
        for j,label in enumerate(list(set(labels))):
            y[labels == label] = true_labels.index(label)


        data = epos.get_data()
        list_session = meta.session.unique()
        cov = Covariances(estimator='lwf')
        matrix = cov.fit_transform(data)
        print(matrix.shape)
        ts = TangentSpace()
        vectors = ts.fit_transform(matrix)
        print(vectors.shape)
        last_session = list_session[-1]
        idx_last = meta.session == last_session
        if len(list_session) > 1:
            X_train = vectors[~idx_last]
            y_train = y[~idx_last]
            X_test = vectors[idx_last]
            y_test = y[idx_last]
            user[k] = [X_train, y_train, X_test, y_test]
            meta_info[k] = {'code': dset.code, 'subject': subject}
        else:
            idx_last = int(vectors.shape[0]*.75)
            ind = np.random.permutation(vectors.shape[0])
            vectors = vectors[ind,:]
            y = y[ind]
            X_train = vectors[:idx_last,:]
            y_train = y[:idx_last]
            X_test = vectors[idx_last:,:]
            y_test = y[idx_last:]
            user[k] = [X_train, y_train, X_test, y_test]
            meta_info[k] = {'code': dset.code, 'subject': subject}

        k += 1

np.savez(filename, user = user, meta_info = meta_info)

# %%
#  last session as test set
#  one subject one client 
#  dataset : dictionary with key = subject 
#            and value = list of train and test session    

import numpy as np
filename ='bci_users54_5c.npz'
data2 = np.load(filename, allow_pickle=True)
dict2 = {key: data2[key] for key in data2.keys() if 'arr_' not in key}
user = dict2['user'].item()
meta_info = dict2['meta_info'].item()
for i in range(len(user)):
    print(meta_info[i])
    print(user[i][0].shape)
    print(user[i][1].shape)
    print(user[i][2].shape)
    print(user[i][3].shape)
    print('----------------')

# %%
