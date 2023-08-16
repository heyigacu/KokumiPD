import sys
import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms

from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix

from rdkit import Chem,DataStructs
from rdkit.Chem import AllChem,Draw,PandasTools
from sklearn.datasets import make_classification
import random
import joblib
from features import RDKFP,E3FP,PLIF, Graph, collate

pwd = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
data_dir = pwd+"/Dataset"
from sklearn.utils import shuffle


from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, EditedNearestNeighbours
from imblearn.combine import SMOTEENN, SMOTETomek
from collections import Counter
import os

def RandomOS(X, y):
    over_sampler = RandomOverSampler()
    X_resampled, y_resampled = over_sampler.fit_resample(X, y) 
    return X_resampled, y_resampled

def SMOTEOS(X, y):
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X, y) 
    return X_resampled, y_resampled

def adasynOS(X,y):
    adasyn = ADASYN()
    X_resampled, y_resampled = adasyn.fit_resample(X, y)
    return X_resampled, y_resampled

def RandomUS(X,y):
    under_sampler = RandomUnderSampler()    
    X_resampled, y_resampled = under_sampler.fit_resample(X, y)
    return X_resampled, y_resampled

def tomek_linksUS(X,y):
    tomek_links = TomekLinks()
    X_resampled, y_resampled = tomek_links.fit_resample(X, y) 
    return X_resampled, y_resampled

def ennUS(X,y):
    enn = EditedNearestNeighbours()
    X_resampled, y_resampled = enn.fit_resample(X, y) 
    return X_resampled, y_resampled

def smote_ennCS(X,y):
    smote_enn = SMOTEENN()
    X_resampled, y_resampled = smote_enn.fit_resample(X, y) 
    return X_resampled, y_resampled
 
def smote_tomekCS(X,y):   
    smote_tomek = SMOTETomek()
    X_resampled, y_resampled = smote_tomek.fit_resample(X, y)  
    return X_resampled, y_resampled

def load_data_kfold_batchsize(tuple_ls, batchsize, graph=False, drop_last=False):
    """
    args:
        ls: [(feature,label)]
        batchsize: int
    """
    if not graph:
        feature, label = tuple_ls[0]
        n_feats = feature.shape[0]
        features, labels = list(zip(*tuple_ls))
        skf = StratifiedKFold(n_splits=5,shuffle=True)
        kfolds=[]
        for train_idxs,val_idxs in skf.split(features, labels):
            trains = [tuple_ls[index] for index in train_idxs]
            trains = DataLoader(trains, batch_size=batchsize, shuffle=True, collate_fn=None, drop_last=drop_last)
            vals = [tuple_ls[index] for index in val_idxs]
            vals = DataLoader(vals,batch_size=len(vals), shuffle=True,)
            kfolds.append((trains,vals))
        return n_feats, kfolds
    else:
        features, labels = list(zip(*tuple_ls))
        skf = StratifiedKFold(n_splits=5,shuffle=True)
        kfolds=[]
        for train_idxs,val_idxs in skf.split(features, labels):
            trains = [tuple_ls[index] for index in train_idxs]
            trains = DataLoader(trains, batch_size=batchsize, shuffle=True, collate_fn=collate, drop_last=drop_last)
            vals = [tuple_ls[index] for index in val_idxs]
            vals = DataLoader(vals,batch_size=len(vals), shuffle=True, collate_fn=collate, drop_last=drop_last)
            kfolds.append((trains,vals))
        return kfolds

def load_data_kfold_notorch(tuple_ls):
    """
    args:
        ls: [(feature,label)]
    """
    random.shuffle(tuple_ls)
    feature,label = tuple_ls[0]
    n_feats = feature.shape[0]
    features, labels = list(zip(*tuple_ls))
    skf = StratifiedKFold(n_splits=5,shuffle=True)
    kfolds=[]
    for train_idxs,val_idxs in skf.split(features, labels):
        trains = [tuple_ls[index] for index in train_idxs]
        vals = [tuple_ls[index] for index in val_idxs]
        kfolds.append((trains,vals))
    return n_feats, kfolds

def load_data_all_batchsize(tuple_ls, batchsize, graph=False, drop_last=False):
    """
    args:
        ls: [(feature,label)]
        batchsize: int
    """
    if not graph:
        feature,label = tuple_ls[0]
        n_feats = feature.shape[0]
        all = DataLoader(tuple_ls, batch_size=batchsize, shuffle=True, collate_fn=None, drop_last=drop_last)
        return n_feats, all
    else:
        return DataLoader(tuple_ls,batch_size=batchsize, shuffle=True, collate_fn=collate, drop_last=drop_last)

def load_data_all_notorch(tuple_ls):
    """
    args:
        ls: [(feature,label)]
    """
    feature,label = tuple_ls[0]
    n_feats = feature.shape[0]
    random.shuffle(tuple_ls)
    return n_feats, tuple_ls


def RandomSample4Multi(path='/home/hy/Documents/Project/KokumiPD/Feature/Image/all_images.csv'):
    df = pd.read_csv(path,sep='\t',header=0)
    balanced_df = df.query('Label==0')
    num = len(balanced_df)
    names = ['Kokumi','Umami','Bitter','Sweet','Salty','Sour','Tasteless']
    for name in names:
        if name != 'Kokumi':
            df_ = df.query('Taste==@name')
            df_ = shuffle(df_)
            df_ = df_.sample(frac=num/len(df_), random_state=1, replace=True)
            balanced_df = pd.concat([balanced_df, df_])
    return balanced_df

def RandomSample4KNK(path='/home/hy/Documents/Project/KokumiPD/Feature/Image/all_images.csv'):
    df = pd.read_csv(path,sep='\t',header=0)
    balanced_df = df.query('Label==0')
    num = len(balanced_df)*1.5
    df_ = df.replace({"Label":{2:1,3:1,4:1,5:1,6:1}})
    df_ = df_.query('Label==1')
    df_ = shuffle(df_)
    df_ = df_.sample(frac=num/len(df_), random_state=1, replace=True)
    balanced_df = pd.concat([balanced_df, df_])
    return balanced_df

def load_six_flavours(path, featurizer=RDKFP, if_all=False, if_torch=False, batchsize=64, used_idx=True, graph=False, drop_last=False):
    """
    args:
        featurizer: function name of featurizer, return a ndarray
        all: bool
        batchsize: int
    """
    df = RandomSample4Multi(path)
    # df = pd.read_csv(path,sep='\t',header=0)
    print(df['Label'].value_counts())
    smileses = df['Smiles']
    labels = df['Label'].astype(int)
    if not used_idx:
        features = [featurizer(smiles) for smiles in smileses]
    else:
        idxs = df['Idx']
        features = [featurizer(idx) for idx in idxs]
    # features, labels =  smote_tomekCS(np.array(features),labels)
    tuple_ls = list(zip(features, labels))
    print(Counter(labels))
    if if_all:
        if if_torch:
            return load_data_all_batchsize(tuple_ls, batchsize, graph, drop_last)
        else:
            return load_data_all_notorch(tuple_ls)
    else:
        if if_torch:
            return load_data_kfold_batchsize(tuple_ls, batchsize, graph, drop_last)
        else:
            return load_data_kfold_notorch(tuple_ls)
        
def load_kokumi_nonkokumi(path, featurizer, if_all, if_torch, batchsize, used_idx=True, graph=False, drop_last=True):
    """
    args:
        featurizer: function name of featurizer, return a ndarray
        all: bool
        batchsize: int
    return:
        all: [(batch_features,_batch_labels)]
        skfolds: [(train_loader, val_loader)]5
                train_loader: [(batch_features, _batch_labels)]
                val_loader: (features, labels)
    """

    df = RandomSample4KNK(path)
    # df = pd.read_csv(path,sep='\t',header=0)
    # df = df.replace({"Label":{2:1,3:1,4:1,5:1,6:1}})
    print(df['Label'].value_counts())
    smileses = df['Smiles']
    idxs = df['Idx']
    labels = df['Label'].astype(int)
    if not used_idx:
        features = [featurizer(smiles) for smiles in smileses]
    else:
        features = [featurizer(idx) for idx in idxs]
    # features,labels  = SMOTEOS(np.array(features),labels)
    tuple_ls =  list(zip(features, labels))
    print(Counter(labels))
    if if_all:
        if if_torch:
            return load_data_all_batchsize(tuple_ls, batchsize, graph, drop_last)
        else:
            return load_data_all_notorch(tuple_ls)
    else:
        if if_torch:
            return load_data_kfold_batchsize(tuple_ls, batchsize, graph, drop_last)
        else:
            return load_data_kfold_notorch(tuple_ls)
