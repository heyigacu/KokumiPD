import argparse
import torch
import joblib
import os
from models import WeavePredictor
from features import RDKFP_smiles,Graph_smiles,collate_smiles
import numpy as np
from torch.utils.data.dataloader import DataLoader
from models import *
import torch.nn.functional as F
import pandas as pd

work_dir = os.path.abspath(os.path.dirname(__file__))


parser = argparse.ArgumentParser(description='kokumi and flavous predictor')
parser.add_argument("-t", "--type", type=int, choices=[0,1], default=0,
                    help="0 is Graph Neural Network (GNN) for predict kukumi/non-kokumi, 1 is Support Vector Machine (SVM) with 2D-fingerprint RDKFP as input"
                    )
parser.add_argument("-i", "--file", type=str, default=work_dir+'/test_smiles.csv', help="input smiles file, don't have a header, only a column smiles")
parser.add_argument("-o", "--out", type=str, default=work_dir+'/result.csv',help="output file")
args = parser.parse_args()


work_dir = os.path.abspath(os.path.dirname(__file__))

smileses = []
with open(args.file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        smileses.append(line.strip())

def svmmulti(smileses):
    clf_svmsix = joblib.load(work_dir+'/six_rdkfp_svm.pkl')
    # check smiles
    total = []
    for smiles in smileses:
        try:
            features = np.array([RDKFP_smiles(smiles.strip())])
            rst = clf_svmsix.predict_proba(features)[0]
            labels = ['Kokumi','Umami','Bitter','Sweet','Salty','Sour','Tasteless']
            string = labels[rst.argmax()]
            ls = []
            ls.append(string)
            for value in rst:
                ls.append('{:.4f}'.format(value))
            total.append(ls)
        except:
            total.append(['error smiles', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan'])

    return total

def gnn(smiles):
    model_gnn = WeavePredictor(node_in_feats=74, edge_in_feats=12, n_tasks=2)
    state_dict = torch.load(os.path.join(work_dir,'/kn_gnn.pth'))
    model_gnn.load_state_dict(state_dict)
    model_gnn.eval()
    # check smiles
    total = []
    for smiles in smileses:
        try:
            for i in list(DataLoader([Graph_smiles(smiles)], batch_size=1, shuffle=False, collate_fn=collate_smiles, drop_last=False)):
                graphs = i
            rst = model_gnn(graphs, graphs.ndata.pop('h'), graphs.edata.pop('e'))
            rst =  F.softmax(rst,dim=1).detach().numpy()[0]
            labels = ['Kokumi','Non-Kokuimi']
            string = labels[rst.argmax()]
            ls = []
            ls.append(string)
            for value in rst:
                ls.append('{:.4f}'.format(value))
            total.append(ls)
        except:
            total.append(['error smiles', 'nan', 'nan'])
    return total

if args.type == 0:
    total = gnn(smileses)
    df = pd.DataFrame(total)
    df.columns = ['Taste','Kokumi','Non-Kokuimi']
    df.insert(0,'Smiles',smileses)
    df.to_csv(args.out,index=False,header=True,sep='\t')
else:
    total = svmmulti(smileses)
    df = pd.DataFrame(total)
    df.columns = ['Taste','Kokumi','Umami','Bitter','Sweet','Salty','Sour','Tasteless']
    df.insert(0,'Smiles',smileses)
    df.to_csv(args.out,index=False,header=True,sep='\t')