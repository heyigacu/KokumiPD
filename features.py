import sys
import os
import dgl
from dgllife.utils import mol_to_bigraph,smiles_to_bigraph
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer
import torch
import numpy as np
import pandas as pd
from rdkit import Chem
from e3fp.pipeline import fprints_from_mol, confs_from_smiles 
from PIL import Image
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms
from rdkit.Chem import AllChem,Draw,PandasTools



def E3FP(mol2_path):

    mol = Chem.MolFromMol2File(mol2_path)
    fprint_params = {'bits': 4096,'first':1}
    fprints = fprints_from_mol(mol, fprint_params=fprint_params)
    fp = fprints[0]
    return fp.to_vector(sparse=False).astype(int)

def arr2bitstring(arr):
    return ''.join(np.binary_repr(x, width=1) for x in arr)


def bitstring2arr(bitstring):
    bit_ls = list(str(bitstring))
    arr = []
    for i in bit_ls:
        arr.append(int(i))
    return np.array(arr)

def RDKFP_smiles(smiles):
    return np.array(Chem.RDKFingerprint(Chem.MolFromSmiles(smiles)))

def RDKFP(idx):
    df = pd.read_csv('/home/hy/Documents/Project/KokumiPD/Feature/RDKFP/all_rdkfp.csv',header=0,sep='\t')
    bitstring = df[df['Idx'] == idx]['RDKFP'].values[0]
    return bitstring2arr(bitstring)

def E3FP(idx):
    df = pd.read_csv('/home/hy/Documents/Project/KokumiPD/Feature/Mol2/check_e3fp.csv',header=0,sep='\t')
    bitstring = df[df['Idx'] == idx]['E3FP'].values[0]
    return bitstring2arr(bitstring)

def PLIF(idx):
    df = pd.read_csv('/home/hy/Documents/Project/KokumiPD/Feature/Docking/check_plif_hippos.csv',header=0,sep='\t')
    bitstring = df[df['Idx'] == idx]['PlifHippos'].values[0]
    return bitstring2arr(bitstring)

def collate(sample):
    graphs, labels = map(list,zip(*sample))
    batched_graph = dgl.batch(graphs)
    batched_graph.set_n_initializer(dgl.init.zero_initializer)
    batched_graph.set_e_initializer(dgl.init.zero_initializer)
    return batched_graph, torch.tensor(labels)

def collate_smiles(graphs):
    batched_graph = dgl.batch(graphs)
    batched_graph.set_n_initializer(dgl.init.zero_initializer)
    batched_graph.set_e_initializer(dgl.init.zero_initializer)
    return batched_graph

def collate_smiles_graph(sample):
    graphs, smiles = map(list,zip(*sample))
    batched_graph = dgl.batch(graphs)
    batched_graph.set_n_initializer(dgl.init.zero_initializer)
    batched_graph.set_e_initializer(dgl.init.zero_initializer)
    return batched_graph, smiles

def Graph(idx):
    df = pd.read_csv('/home/hy/Documents/Project/KokumiPD/Dataset/All.csv',header=0,sep='\t')
    smiles = df[df['Idx'] == idx]['Smiles'].values[0]
    return mol_to_bigraph(Chem.MolFromSmiles(smiles), node_featurizer=CanonicalAtomFeaturizer(atom_data_field='h'), edge_featurizer=CanonicalBondFeaturizer(bond_data_field='e'))

def Graph_smiles(smiles):
    return mol_to_bigraph(Chem.MolFromSmiles(smiles), node_featurizer=CanonicalAtomFeaturizer(atom_data_field='h'), edge_featurizer=CanonicalBondFeaturizer(bond_data_field='e'))

def data_augmentation(tuple_ls, transform):
    return [(transform(Image.open(img_path).convert('RGB')),label) for (img_path,label) in tuple_ls]

def cnn_prepare(tuple_ls):
    # data_augmentation
    original_transform = transforms.Compose([transforms.ToTensor()])
    
    flip_transform = transforms.Compose([
        transforms.RandomChoice([
            transforms.RandomHorizontalFlip(p=1), 
            transforms.RandomVerticalFlip(p=1),
            transforms.Compose([
                transforms.RandomHorizontalFlip(p=1), 
                transforms.RandomHorizontalFlip(p=1)
            ]),      
        ]),
        transforms.ToTensor(),
        ])
    rotate_transform = transforms.Compose([
        transforms.RandomChoice([
        transforms.RandomRotation(degrees=(90, 90)) ,
        transforms.RandomRotation(degrees=(180, 180)),
        transforms.RandomRotation(degrees=(270, 270)),
        ]),
        transforms.ToTensor(),
    ])
    zoom_transform = transforms.Compose([
        transforms.RandomChoice([
            transforms.Compose([
                transforms.Resize((74, 74)), 
                transforms.Pad(padding=10, fill=(255,255,255), padding_mode="constant"),
            ]),
            transforms.Compose([
                transforms.Resize((54, 54)), 
                transforms.Pad(padding=20, fill=(255,255,255), padding_mode="constant"),
            ]),
            transforms.Compose([
                transforms.Resize((34, 34)), 
                transforms.Pad(padding=30, fill=(255,255,255), padding_mode="constant"),
            ]),    
        ]),
        transforms.ToTensor(),
    ])

    original = data_augmentation(tuple_ls,original_transform)

    flip_tuple_ls= [tuple_ls[i] for i in [np.random.randint(0,len(tuple_ls)) for i in range(int(len(tuple_ls)/2))]]
    flip = data_augmentation(flip_tuple_ls,flip_transform)

    rotate_tuple_ls= [tuple_ls[i] for i in [np.random.randint(0,len(tuple_ls)) for i in range(int(len(tuple_ls)/2))]]
    rotate = data_augmentation(rotate_tuple_ls,rotate_transform)

    zoom_tuple_ls= [tuple_ls[i] for i in [np.random.randint(0,len(tuple_ls)) for i in range(int(len(tuple_ls)/2))]]
    zoom = data_augmentation(zoom_tuple_ls,zoom_transform)

    return original+flip+rotate+zoom

def image_featurizer(df):
    imgs = df['Image']
    labels = df['Label']
    tuple_ls = list(zip(imgs,labels))
    original = cnn_prepare(tuple_ls)
    return original


