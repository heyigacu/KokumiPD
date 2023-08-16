"""
written by Yi He, July 2023
"""
import sys
import os
import torch
import torch.optim as optim
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import KFold
import numpy as np
from models import MLP
from models import AdaBoost,RF,SVM,train_gnn,train_mlp
from utils import *
from load_data import *
from features import *
from rdkit import Chem
import joblib

work_dir = os.path.abspath(os.path.dirname(__file__))
batchsize_kn = 128
drop_last=True
ls_kn = []

########################
def train_kn_rdkfp_rfsvm_skfolds():
    n_rdkfp_feats, kn_rdkfp_skfolds =  load_kokumi_nonkokumi('/home/hy/Documents/Project/KokumiPD/Dataset/All.csv', featurizer=RDKFP, if_all=False, if_torch=False, batchsize=batchsize_kn, drop_last=drop_last)
    n_rdkfp_feats, kn_rdkfp_all =  load_kokumi_nonkokumi('/home/hy/Documents/Project/KokumiPD/Dataset/All.csv',featurizer=RDKFP, if_all=True, if_torch=False, batchsize=batchsize_kn, drop_last=drop_last)
    kfolds=kn_rdkfp_skfolds
    all=kn_rdkfp_all
    rf = RF()
    rf_metrics = rf.train(bi_classify_metrics,kfolds,all)
    joblib.dump(rf.clf, work_dir+'/models/kn_rdkfp_rf.pkl')
    rf =  joblib.load(work_dir+'/models/kn_rdkfp_rf.pkl')
    print(np.around(np.array(rf_metrics),2))
    ls_kn.append(np.around(np.array(rf_metrics),2))

    svm = SVM()
    svm_metrics = svm.train(bi_classify_metrics,kfolds,all)
    joblib.dump(svm.clf, work_dir+'/models/kn_rdkfp_svm.pkl')
    svm =  joblib.load(work_dir+'/models/kn_rdkfp_svm.pkl')
    print(np.around(np.array(svm_metrics),2))
    ls_kn.append(np.around(np.array(svm_metrics),2))

def train_kn_rdkfp_mlp_skfolds():
    n_feats, kn_rdkfp_torch_skfolds =  load_kokumi_nonkokumi('/home/hy/Documents/Project/KokumiPD/Dataset/All.csv', featurizer=RDKFP, if_all=False, if_torch=True, batchsize=batchsize_kn, drop_last=drop_last)
    n_feats, kn_rdkfp_torch_all =  load_kokumi_nonkokumi('/home/hy/Documents/Project/KokumiPD/Dataset/All.csv',featurizer=RDKFP, if_all=True, if_torch=True, batchsize=batchsize_kn, drop_last=drop_last)
    kfolds=kn_rdkfp_torch_skfolds
    all=kn_rdkfp_torch_all
    mlp_metrics = train_mlp.train_kfolds(n_feats, 2, kfolds, bi_classify_metrics, 12, 500, save_folder='/home/hy/Documents/Project/KokumiPD/Predictor/models/',save_name='kn_rdkfp_mlp.pth',patience=8)
    train_mlp.train_all(n_feats, 2, all, 500, save_folder='/home/hy/Documents/Project/KokumiPD/Predictor/models/',save_name='kn_rdkfp_mlp.pth',patience=8)
    print(np.around(np.array(mlp_metrics),2))
    ls_kn.append(np.around(np.array(mlp_metrics),2))


##########################
def train_kn_e3fp_rfsvm_skfolds():
    n_e3fp_feats, kn_e3fp_skfolds =  load_kokumi_nonkokumi('/home/hy/Documents/Project/KokumiPD/Feature/Mol2/check_e3fp.csv', featurizer=E3FP, if_all=False, if_torch=False, batchsize=batchsize_kn, drop_last=drop_last)
    n_e3fp_feats, kn_e3fp_all =  load_kokumi_nonkokumi('/home/hy/Documents/Project/KokumiPD/Feature/Mol2/check_e3fp.csv',featurizer=E3FP, if_all=True, if_torch=False, batchsize=batchsize_kn, drop_last=drop_last)
    kfolds=kn_e3fp_skfolds
    all=kn_e3fp_all
    rf = RF()
    rf_metrics = rf.train(bi_classify_metrics,kfolds,all)
    joblib.dump(rf.clf, work_dir+'/models/kn_e3fp_rf.pkl')
    rf =  joblib.load(work_dir+'/models/kn_e3fp_rf.pkl')
    print(np.around(np.array(rf_metrics),2))
    ls_kn.append(np.around(np.array(rf_metrics),2))

    svm = SVM()
    svm_metrics = svm.train(bi_classify_metrics,kfolds,all)
    joblib.dump(svm.clf, work_dir+'/models/kn_e3fp_svm.pkl')
    svm =  joblib.load(work_dir+'/models/kn_e3fp_svm.pkl')
    print(np.around(np.array(svm_metrics),2))
    ls_kn.append(np.around(np.array(svm_metrics),2))

def train_kn_e3fp_mlp_skfolds():
    n_feats, kn_e3fp_torch_skfolds =  load_kokumi_nonkokumi('/home/hy/Documents/Project/KokumiPD/Feature/Mol2/check_e3fp.csv', featurizer=E3FP, if_all=False, if_torch=True, batchsize=batchsize_kn, drop_last=drop_last)
    n_feats, kn_e3fp_torch_all =  load_kokumi_nonkokumi('/home/hy/Documents/Project/KokumiPD/Feature/Mol2/check_e3fp.csv',featurizer=E3FP, if_all=True, if_torch=True, batchsize=batchsize_kn, drop_last=drop_last)
    kfolds=kn_e3fp_torch_skfolds
    all=kn_e3fp_torch_all
    mlp_metrics = train_mlp.train_kfolds(n_feats, 2, kfolds, bi_classify_metrics, 12, 500, save_folder='/home/hy/Documents/Project/KokumiPD/Predictor/models/',save_name='kn_e3fp_mlp.pth',patience=8)
    train_mlp.train_all(n_feats, 2, all, 500, save_folder='/home/hy/Documents/Project/KokumiPD/Predictor/models/',save_name='kn_e3fp_mlp.pth',patience=8)
    print(np.around(np.array(mlp_metrics),2))
    ls_kn.append(np.around(np.array(mlp_metrics),2))

##########################
def train_kn_plif_rfsvm_skfolds():
    n_plif_feats, kn_plif_skfolds =  load_kokumi_nonkokumi('/home/hy/Documents/Project/KokumiPD/Feature/Docking/check_plif_hippos.csv', featurizer=PLIF, if_all=False, if_torch=False, batchsize=batchsize_kn, drop_last=drop_last)
    n_plif_feats, kn_plif_all =  load_kokumi_nonkokumi('/home/hy/Documents/Project/KokumiPD/Feature/Docking/check_plif_hippos.csv',featurizer=PLIF, if_all=True, if_torch=False, batchsize=batchsize_kn, drop_last=drop_last)
    kfolds=kn_plif_skfolds
    all=kn_plif_all
    rf = RF()
    rf_metrics = rf.train(bi_classify_metrics,kfolds,all)
    joblib.dump(rf.clf, work_dir+'/models/kn_plif_rf.pkl')
    rf =  joblib.load(work_dir+'/models/kn_plif_rf.pkl')
    print(np.around(np.array(rf_metrics),2))
    ls_kn.append(np.around(np.array(rf_metrics),2))

    svm = SVM()
    svm_metrics = svm.train(bi_classify_metrics,kfolds,all)
    joblib.dump(svm.clf, work_dir+'/models/kn_plif_svm.pkl')
    svm =  joblib.load(work_dir+'/models/kn_plif_svm.pkl')
    print(np.around(np.array(svm_metrics),2))
    ls_kn.append(np.around(np.array(svm_metrics),2))

##########################
def train_kn_plif_mlp_skfolds():
    n_feats, kn_plif_torch_skfolds =  load_kokumi_nonkokumi('/home/hy/Documents/Project/KokumiPD/Feature/Docking/check_plif_hippos.csv', featurizer=PLIF, if_all=False, if_torch=True, batchsize=batchsize_kn, drop_last=drop_last)
    n_feats, kn_plif_torch_all =  load_kokumi_nonkokumi('/home/hy/Documents/Project/KokumiPD/Feature/Docking/check_plif_hippos.csv',featurizer=PLIF, if_all=True, if_torch=True, batchsize=batchsize_kn, drop_last=drop_last)
    kfolds=kn_plif_torch_skfolds
    all=kn_plif_torch_all
    mlp_metrics = train_mlp.train_kfolds(n_feats, 2, kfolds, bi_classify_metrics, 12, 500, save_folder='/home/hy/Documents/Project/KokumiPD/Predictor/models/',save_name='kn_plif_mlp.pth',patience=8)
    train_mlp.train_all(n_feats, 2, all, 500, save_folder='/home/hy/Documents/Project/KokumiPD/Predictor/models/',save_name='kn_plif_mlp.pth',patience=8)
    print(np.around(np.array(mlp_metrics),2))
    ls_kn.append(np.around(np.array(mlp_metrics),2))

# #######################
def train_kn_gnn_skfolds():
    kn_g_torch_skfolds =  load_kokumi_nonkokumi(path='/home/hy/Documents/Project/KokumiPD/Dataset/All.csv', featurizer=Graph, if_all=False, if_torch=True, batchsize=batchsize_kn,  graph=True, drop_last=drop_last)
    kn_g_torch_all =  load_kokumi_nonkokumi(path='/home/hy/Documents/Project/KokumiPD/Dataset/All.csv',featurizer=Graph, if_all=True, if_torch=True, batchsize=batchsize_kn, graph=True, drop_last=drop_last)
    kfolds=kn_g_torch_skfolds
    all=kn_g_torch_all
    gnn_metrics = train_gnn.train_kfolds(n_atom_feat=74, n_bond_feat=12, n_classes=2, skfolds=kfolds, classify_metrics=bi_classify_metrics, n_metrics=12, max_epochs=500,save_name='kn_gnn.pth',patience=15)
    train_gnn().train_all(74, 12, 2, all, 500, save_folder='/home/hy/Documents/Project/KokumiPD/Predictor/models/',save_name='kn_gnn.pth',patience=15)
    print(np.around(np.array(gnn_metrics),2))
    ls_kn.append(np.around(np.array(gnn_metrics),2))

train_kn_rdkfp_rfsvm_skfolds()
train_kn_rdkfp_mlp_skfolds()
train_kn_e3fp_rfsvm_skfolds()
train_kn_e3fp_mlp_skfolds()
train_kn_plif_rfsvm_skfolds()
train_kn_plif_mlp_skfolds()
train_kn_gnn_skfolds()
print(ls_kn)
pd.DataFrame(ls_kn).to_csv('/home/hy/Documents/Project/KokumiPD/Predictor/models/kn.txt',sep='\t',index=False,header=False)


# ##########################
# ##########################
# ##########################
# ##########################
# ##########################
# ##########################
# ##########################

batchsize_six = 128
drop_last = True
ls_six = []
##########################
def train_six_rdkfp_rfsvm_skfolds():
    n_rdkfp_feats, six_rdkfp_skfolds =  load_six_flavours('/home/hy/Documents/Project/KokumiPD/Dataset/All.csv', featurizer=RDKFP, if_all=False, if_torch=False, batchsize=batchsize_six, drop_last=drop_last)
    n_rdkfp_feats, six_rdkfp_all =  load_six_flavours('/home/hy/Documents/Project/KokumiPD/Dataset/All.csv',featurizer=RDKFP, if_all=True, if_torch=False, batchsize=batchsize_six, drop_last=drop_last)
    kfolds=six_rdkfp_skfolds
    all=six_rdkfp_all
    rf = RF()
    rf_metrics = rf.train(multi_classify_metrics,kfolds,all)
    joblib.dump(rf.clf, work_dir+'/models/six_rdkfp_rf.pkl')
    rf =  joblib.load(work_dir+'/models/six_rdkfp_rf.pkl')
    print(np.around(np.array(rf_metrics),2))
    ls_six.append(np.around(np.array(rf_metrics),2))

    svm = SVM()
    svm_metrics = svm.train(multi_classify_metrics,kfolds,all)
    joblib.dump(svm.clf, work_dir+'/models/six_rdkfp_svm.pkl')
    svm =  joblib.load(work_dir+'/models/six_rdkfp_svm.pkl')
    print(np.around(np.array(svm_metrics),2))
    ls_six.append(np.around(np.array(svm_metrics),2))

def train_six_rdkfp_mlp_skfolds():
    n_feats, six_rdkfp_torch_skfolds =  load_six_flavours('/home/hy/Documents/Project/KokumiPD/Dataset/All.csv', featurizer=RDKFP, if_all=False, if_torch=True, batchsize=batchsize_six, drop_last=drop_last)
    n_feats, six_rdkfp_torch_all =  load_six_flavours('/home/hy/Documents/Project/KokumiPD/Dataset/All.csv',featurizer=RDKFP, if_all=True, if_torch=True, batchsize=batchsize_six, drop_last=drop_last)
    kfolds=six_rdkfp_torch_skfolds
    all=six_rdkfp_torch_all
    mlp_metrics = train_mlp.train_kfolds(n_feats, 7, kfolds, multi_classify_metrics, 12, 500, save_folder='/home/hy/Documents/Project/KokumiPD/Predictor/models/',save_name='six_rdkfp_mlp.pth',patience=8)
    train_mlp.train_all(n_feats, 7, all, 500, save_folder='/home/hy/Documents/Project/KokumiPD/Predictor/models/',save_name='six_rdkfp_mlp.pth',patience=8)
    print(np.around(np.array(mlp_metrics),2))
    ls_six.append(np.around(np.array(mlp_metrics),2))

##########################
def train_six_e3fp_rfsvm_skfolds():
    n_e3fp_feats, six_e3fp_skfolds =  load_six_flavours('/home/hy/Documents/Project/KokumiPD/Feature/Mol2/check_e3fp.csv', featurizer=E3FP, if_all=False, if_torch=False, batchsize=batchsize_six, drop_last=drop_last)
    n_e3fp_feats, six_e3fp_all =  load_six_flavours('/home/hy/Documents/Project/KokumiPD/Feature/Mol2/check_e3fp.csv',featurizer=E3FP, if_all=True, if_torch=False, batchsize=batchsize_six, drop_last=drop_last)
    kfolds=six_e3fp_skfolds
    all=six_e3fp_all
    rf = RF()
    rf_metrics = rf.train(multi_classify_metrics,kfolds,all)
    joblib.dump(rf.clf, work_dir+'/models/six_e3fp_rf.pkl')
    rf =  joblib.load(work_dir+'/models/six_e3fp_rf.pkl')
    print(np.around(np.array(rf_metrics),2))
    ls_six.append(np.around(np.array(rf_metrics),2))

    svm = SVM()
    svm_metrics = svm.train(multi_classify_metrics,kfolds,all)
    joblib.dump(svm.clf, work_dir+'/models/six_e3fp_svm.pkl')
    svm =  joblib.load(work_dir+'/models/six_e3fp_svm.pkl')
    print(np.around(np.array(svm_metrics),2))
    ls_six.append(np.around(np.array(svm_metrics),2))

def train_six_e3fp_mlp_skfolds():
    n_feats, six_e3fp_torch_skfolds =  load_six_flavours('/home/hy/Documents/Project/KokumiPD/Feature/Mol2/check_e3fp.csv', featurizer=E3FP, if_all=False, if_torch=True, batchsize=batchsize_six, drop_last=drop_last)
    n_feats, six_e3fp_torch_all =  load_six_flavours('/home/hy/Documents/Project/KokumiPD/Feature/Mol2/check_e3fp.csv',featurizer=E3FP, if_all=True, if_torch=True, batchsize=batchsize_six, drop_last=drop_last)
    kfolds=six_e3fp_torch_skfolds
    all=six_e3fp_torch_all
    mlp_metrics = train_mlp.train_kfolds(n_feats, 7, kfolds, multi_classify_metrics, 12, 500, save_folder='/home/hy/Documents/Project/KokumiPD/Predictor/models/',save_name='six_e3fp_mlp.pth',patience=8)
    train_mlp.train_all(n_feats, 7, all, 500, save_folder='/home/hy/Documents/Project/KokumiPD/Predictor/models/',save_name='six_e3fp_mlp.pth',patience=8)
    print(np.around(np.array(mlp_metrics),2))
    ls_six.append(np.around(np.array(mlp_metrics),2))

#########################
def train_six_plif_rfsvm_skfolds():
    n_plif_feats, six_plif_skfolds =  load_six_flavours('/home/hy/Documents/Project/KokumiPD/Feature/Docking/check_plif_hippos.csv', featurizer=PLIF, if_all=False, if_torch=False, batchsize=batchsize_six, drop_last=drop_last)
    n_plif_feats, six_plif_all =  load_six_flavours('/home/hy/Documents/Project/KokumiPD/Feature/Docking/check_plif_hippos.csv',featurizer=PLIF, if_all=True, if_torch=False, batchsize=batchsize_six, drop_last=drop_last)
    kfolds=six_plif_skfolds
    all=six_plif_all
    rf = RF()
    rf_metrics = rf.train(multi_classify_metrics,kfolds,all)
    joblib.dump(rf.clf, work_dir+'/models/six_plif_rf.pkl')
    rf =  joblib.load(work_dir+'/models/six_plif_rf.pkl')
    print(np.around(np.array(rf_metrics),2))
    ls_six.append(np.around(np.array(rf_metrics),2))

    svm = SVM()
    svm_metrics = svm.train(multi_classify_metrics,kfolds,all)
    joblib.dump(svm.clf, work_dir+'/models/six_plif_svm.pkl')
    svm =  joblib.load(work_dir+'/models/six_plif_svm.pkl')
    print(np.around(np.array(svm_metrics),2))
    ls_six.append(np.around(np.array(svm_metrics),2))

def train_six_plif_mlp_skfolds():
    n_feats, six_plif_torch_skfolds =  load_six_flavours('/home/hy/Documents/Project/KokumiPD/Feature/Docking/check_plif_hippos.csv', featurizer=PLIF, if_all=False, if_torch=True, batchsize=batchsize_six, drop_last=drop_last)
    n_feats, six_plif_torch_all =  load_six_flavours('/home/hy/Documents/Project/KokumiPD/Feature/Docking/check_plif_hippos.csv',featurizer=PLIF, if_all=True, if_torch=True, batchsize=batchsize_six, drop_last=drop_last)
    kfolds=six_plif_torch_skfolds
    all=six_plif_torch_all
    mlp_metrics = train_mlp.train_kfolds(n_feats, 7, kfolds, multi_classify_metrics, 12, 500, save_folder='/home/hy/Documents/Project/KokumiPD/Predictor/models/',save_name='six_plif_mlp.pth',patience=8)
    train_mlp.train_all(n_feats, 7, all, 500, save_folder='/home/hy/Documents/Project/KokumiPD/Predictor/models/',save_name='six_plif_mlp.pth',patience=8)
    print(np.around(np.array(mlp_metrics),2))
    ls_six.append(np.around(np.array(mlp_metrics),2))

########################
def train_six_gnn_skfolds():
    six_g_torch_skfolds =  load_six_flavours(path='/home/hy/Documents/Project/KokumiPD/Dataset/All.csv', featurizer=Graph, if_all=False, if_torch=True, batchsize=batchsize_six,  graph=True, drop_last=drop_last)
    six_g_torch_all =  load_six_flavours(path='/home/hy/Documents/Project/KokumiPD/Dataset/All.csv',featurizer=Graph, if_all=True, if_torch=True, batchsize=batchsize_six, graph=True, drop_last=drop_last)
    kfolds=six_g_torch_skfolds
    all=six_g_torch_all
    gnn_metrics = train_gnn.train_kfolds(n_atom_feat=74, 
                                         n_bond_feat=12, 
                                         n_classes=7, 
                                         skfolds=kfolds, 
                                         n_metrics=7, 
                                         classify_metrics=multi_classify_metrics, 
                                         max_epochs=500, 
                                         save_folder='/home/hy/Documents/Project/KokumiPD/Predictor/models/',
                                         save_name='six_gnn.pth',
                                         patience=15)
    train_gnn().train_all(n_atom_feat=74, n_bond_feat=12, n_classes=7, all=all, max_epochs=500, save_folder='/home/hy/Documents/Project/KokumiPD/Predictor/models/',save_name='six_gnn.pth',patience=15)
    print(np.around(np.array(gnn_metrics),2))
    ls_six.append(np.around(np.array(gnn_metrics),2))


train_six_rdkfp_rfsvm_skfolds()
train_six_rdkfp_mlp_skfolds()
train_six_e3fp_rfsvm_skfolds()
train_six_e3fp_mlp_skfolds()
train_six_plif_rfsvm_skfolds()
train_six_plif_mlp_skfolds()
train_six_gnn_skfolds()
print(ls_six)
pd.DataFrame(ls_six).to_csv('/home/hy/Documents/Project/KokumiPD/Predictor/models/six.txt',sep='\t',index=False,header=False)