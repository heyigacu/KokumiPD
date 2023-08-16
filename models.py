import sys
import os
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn import svm
from sklearn.metrics import roc_auc_score,accuracy_score
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,AdaBoostRegressor
from  dgllife.model.gnn import WeaveGNN
from  dgllife.model.readout import WeightedSumAndMax,SumAndMax
from early_stop import EarlyStopping
from utils import acc_auc

work_dir = os.path.abspath(os.path.dirname(__file__))


class AdaBoost():
    def __init__(self) -> None:
        pass
    def train(self, classify_metrics, skfolds, all):
        metrics = []
        aucs = []
        for (train_fold,val_fold) in skfolds:
            train_x,train_y = list(zip(*train_fold))
            train_x,train_y = np.array(list(train_x)),np.array(list(train_y))
            (val_x,val_y) = list(zip(*val_fold))
            val_x,val_y = np.array(list(val_x)),np.array(list(val_y))
            clf = AdaBoostClassifier(algorithm='SAMME.R',learning_rate=0.1, n_estimators=100, random_state=0)
            clf.fit(train_x, train_y)
            preds = clf.predict_proba(val_x)
            metrics.append(classify_metrics(val_y,preds))
            print(accuracy_score(val_y,preds.argmax(-1)))
            aucs.append(roc_auc_score(val_y,preds,multi_class='ovr',average="micro"))
        x, y = list(zip(*all))
        x=list(x)
        y=list(y)
        clf = AdaBoostClassifier(algorithm='SAMME.R',learning_rate=1.0, n_estimators=100, random_state=0)
        clf.fit(x, y)
        self.clf=clf
        return np.array(metrics).mean(0)
    def test(self):
        pass
    def predict(self, features):
        preds = self.clf.predict_proba(features)
        return preds.argmax(-1)

class RF():
    def __init__(self) -> None:
        pass
    def train(self, classify_metrics, skfolds, all):
        metrics = []
        for (train_fold,val_fold) in skfolds:
            train_x,train_y = list(zip(*train_fold))
            train_x,train_y = np.array(list(train_x)),np.array(list(train_y))
            (val_x,val_y) = list(zip(*val_fold))
            val_x,val_y = np.array(list(val_x)),np.array(list(val_y))
            clf = RandomForestClassifier(n_estimators=200, random_state=0)
            clf.fit(train_x, train_y)
            preds = clf.predict_proba(val_x)
            metrics.append(classify_metrics(val_y,preds))
        x, y = list(zip(*all))
        x=list(x)
        y=list(y)
        clf = RandomForestClassifier(n_estimators=200, random_state=0)
        clf.fit(x, y)
        self.clf=clf
        return np.array(metrics).mean(0)
    def test(self):
        pass
    def predict(self, features):
        preds = self.clf.predict_proba(features)
        return preds.argmax(-1)

class SVM():
    def __init__(self) -> None:
        pass
    def train(self, classify_metrics, skfolds, all):
        metrics = []
        for (train_fold,val_fold) in skfolds:
            train_x,train_y = list(zip(*train_fold))
            train_x,train_y = np.array(list(train_x)),np.array(list(train_y))
            (val_x,val_y) = list(zip(*val_fold))
            val_x,val_y = np.array(list(val_x)),np.array(list(val_y))
            clf = svm.SVC(C=50, kernel='rbf',probability=True, decision_function_shape='ovr')
            clf.fit(train_x, train_y)
            preds = clf.predict_proba(val_x)
            metrics.append(classify_metrics(val_y,preds))
        x, y = list(zip(*all))
        x=list(x)
        y=list(y)
        clf = svm.SVC(C=50, kernel='rbf',probability=True, decision_function_shape='ovr')
        clf.fit(x, y)
        self.clf=clf
        return np.array(metrics).mean(0)
    def test(self):
        pass
    def predict(self, features):
        preds = self.clf.predict_proba(features)
        return preds
class MLP(nn.Module):
    def __init__(self, n_feats, n_classes):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_feats, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, n_classes),
        )
        self.criteon = nn.CrossEntropyLoss()
    def forward(self, x):
        x = self.model(x)
        return x


class train_mlp(object):
    @staticmethod
    def train_kfolds(n_feats, n_classes, skfolds, classify_metrics, n_metrics, max_epochs, save_folder='/home/hy/Documents/Project/KokumiPD/Predictor/models/',save_name='test.pth', patience =10):
        print(n_feats)
        train_losses = []
        train_accs = []
        train_aucs = []
        val_losses = []
        val_metrics = []
        for train_loader,val_loader in skfolds:
            if torch.cuda.is_available():
                device = torch.device('cuda:0')
            else:
                device = "cpu"
            model = MLP(n_feats, n_classes).to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            early_stopping = EarlyStopping(save_path=save_folder,save_name=save_name,patience=patience)
            for epoch in range(1, max_epochs+1):
                model.train()
                loss_train = 0.
                auc_train = 0.
                acc_train = 0.
                for batch_idx,(train_features,train_labels) in enumerate(train_loader):
                    features, labels = train_features.to(device), train_labels.to(device)
                    logits = model(features.to(torch.float32))
                    loss = model.criteon(logits, labels)
                    loss_train += loss.detach().item()
                    acc,auc = acc_auc(labels.cpu().numpy(), logits.detach().cpu().numpy())
                    acc_train += acc
                    auc_train += auc
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                loss_train /= (batch_idx+1)
                auc_train /= (batch_idx+1)
                acc_train /= (batch_idx+1)
                if epoch%1 == 0:
                    train_losses.append(loss_train)
                    train_accs.append(acc_train)
                    train_aucs.append(auc_train)
                # metrics_val = np.zeros(n_metrics)
                model.eval()
                with torch.no_grad():
                    for val_features, val_labels in val_loader:
                        features, labels = val_features.to(device), val_labels.to(device)
                        logits = model(features.to(torch.float32))  
                        loss = model.criteon(logits, labels)   
                        loss_val = loss.detach().item()
                        metrics_val = classify_metrics(labels.cpu().numpy(), logits.detach().cpu().numpy())
                if epoch%1 == 0:
                    print('loss_train:',loss_train,'ACC:',acc_train,'AUC:',auc_train, 'loss_val:',loss_val, 'ACC:',metrics_val[2], 'AUC:',metrics_val[-1])
                    pass
                    # val_losses.append(loss_val)
                    # val_metrics.append(metrics_val)
                early_stopping(loss_val, model)
                if early_stopping.early_stop:
                    val_metrics.append(metrics_val)
                    print("Early stopping")
                    break
                else:
                    if epoch == max_epochs:
                        val_metrics.append(metrics_val)      
        return np.array(val_metrics).mean(0)
    @staticmethod
    def train_all(n_feats, n_classes, all, max_epochs, save_folder='/home/hy/Documents/Project/KokumiPD/Predictor/models/',save_name='test.pth', patience =10):
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = "cpu"
        model = MLP(n_feats, n_classes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        early_stopping = EarlyStopping(save_path=save_folder,save_name=save_name,patience=patience)
        rst = []
        for epoch in range(1, max_epochs+1):
            loss_train = 0.
            auc_train = 0.
            acc_train = 0.
            model.train()
            for batch_idx,(train_features,train_labels) in enumerate(all):
                features, labels = train_features.to(device), train_labels.to(device)
                logits = model(features.to(torch.float32))
                loss = model.criteon(logits, labels)
                loss_train += loss.detach().item()
                acc,auc = acc_auc(labels.cpu().numpy(), logits.detach().cpu().numpy())
                acc_train += acc
                auc_train += auc
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_train /= (batch_idx+1)
            auc_train /= (batch_idx+1)
            acc_train /= (batch_idx+1)
            if epoch%1 == 0:
                rst.append(np.array([loss_train, acc_train, auc_train]))
                print('loss:',loss_train,'ACC:',acc_train,'AUC:',auc_train)      
            early_stopping(loss_train, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        print(np.array(rst).shape)           
        np.savetxt(save_folder+'train_mlp.txt', rst)
    

class WeavePredictor(nn.Module):
    def __init__(self,
                 node_in_feats,
                 edge_in_feats,
                 num_gnn_layers=2,
                 gnn_hidden_feats=128,
                 gnn_activation=F.relu,
                 n_tasks=1,
                ):
        super(WeavePredictor, self).__init__()

        self.gnn = WeaveGNN(node_in_feats=node_in_feats,
                            edge_in_feats=edge_in_feats,
                            num_layers=num_gnn_layers,
                            hidden_feats=gnn_hidden_feats,
                            activation=gnn_activation)

        self.readout = WeightedSumAndMax(gnn_hidden_feats)

        self.predict = nn.Sequential(
            nn.Linear(2*gnn_hidden_feats, 64),             
            nn.Linear(64, n_tasks),
        )

    
    def forward(self, g, node_feats, edge_feats):
        """Graph-level regression/soft classification.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_in_feats)
            Input node features. V for the number of nodes.
        edge_feats : float32 tensor of shape (E, edge_in_feats)
            Input edge features. E for the number of edges.

        Returns
        -------
        float32 tensor of shape (G, n_tasks)
            Prediction for the graphs in the batch. G for the number of graphs.
        """
        node_feats = self.gnn(g, node_feats, edge_feats, node_only=True)
        g_feats = self.readout(g, node_feats)
        return self.predict(g_feats)
    
class train_gnn(object):
    def __init__(self) -> None:
        pass
    @staticmethod
    def train_kfolds(n_atom_feat = 74, n_bond_feat = 12, n_classes=2, skfolds=None, classify_metrics=None, n_metrics=7, max_epochs=100, save_folder='/home/hy/Documents/Project/KokumiPD/Predictor/models/',save_name='test.pth',patience=10):
        train_losses = []
        train_accs = []
        train_aucs = []
        val_losses = []
        val_metrics = []
        for train_loader,val_loader in skfolds:
            if torch.cuda.is_available():
                device = torch.device('cuda:0')
            else:
                device = "cpu"
            model = WeavePredictor(node_in_feats=n_atom_feat, edge_in_feats=n_bond_feat, n_tasks=n_classes).to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            early_stopping = EarlyStopping(save_folder,save_name,patience=patience)
            for epoch in range(1, max_epochs+1):
                model.train()
                loss_train = 0.
                auc_train = 0.
                acc_train = 0.
                for batch_idx,(train_graphs,train_labels) in enumerate(train_loader):
                    graphs, labels = train_graphs.to(device), train_labels.to(device)
                    preds = model(graphs, graphs.ndata.pop('h'), graphs.edata.pop('e'))
                    loss = CrossEntropyLoss()(preds, labels)
                    loss_train += loss.detach().item()
                    acc,auc = acc_auc(labels.cpu().numpy(), preds.detach().cpu().numpy())
                    acc_train += acc
                    auc_train += auc
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                loss_train /= (batch_idx+1)
                auc_train /= (batch_idx+1)
                acc_train /= (batch_idx+1)
                if epoch%1 == 0:
                    print('loss:',loss_train,'ACC:',acc_train,'AUC:',auc_train)
                    train_losses.append(loss_train)
                    train_accs.append(acc_train)
                    train_aucs.append(auc_train)
                # metrics_val = np.zeros(n_metrics)
                model.eval()
                with torch.no_grad():
                    for val_graphs, val_labels in val_loader:
                        graphs, labels = val_graphs.to(device), val_labels.to(device)
                        preds = model(graphs, graphs.ndata.pop('h'), graphs.edata.pop('e'))
                        loss = CrossEntropyLoss()(preds, labels)
                        loss_val = loss.detach().item()
                        metrics_val = classify_metrics(labels.cpu().numpy(), preds.detach().cpu().numpy())
                if epoch%1 == 0:
                    pass
                    # val_losses.append(loss_val)
                    # val_metrics.append(metrics_val)
                early_stopping(loss_val, model)
                if early_stopping.early_stop:
                    val_metrics.append(metrics_val)
                    print("Early stopping")
                    break
                else:
                    if epoch == max_epochs:
                        val_metrics.append(metrics_val)
        return np.array(val_metrics).mean(0)
    def train_all(self, n_atom_feat, n_bond_feat, n_classes, all, max_epochs, save_folder='/home/hy/Documents/Project/KokumiPD/Predictor/models/',save_name='test.pth',patience=10):
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = "cpu"
        model = WeavePredictor(node_in_feats=n_atom_feat, edge_in_feats=n_bond_feat, n_tasks=n_classes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        early_stopping = EarlyStopping(save_folder,save_name=save_name,patience=patience)
        rst = []
        for epoch in range(1, max_epochs+1):
            loss_train = 0.
            auc_train = 0.
            acc_train = 0.
            model.train()
            for batch_idx,(train_graphs,train_labels) in enumerate(all):
                graphs, labels = train_graphs.to(device), train_labels.to(device)
                logits = model(graphs, graphs.ndata.pop('h'), graphs.edata.pop('e'))
                loss = CrossEntropyLoss()(logits, labels)
                loss_train += loss.detach().item()
                acc,auc = acc_auc(labels.cpu().numpy(), logits.detach().cpu().numpy())
                acc_train += acc
                auc_train += auc
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_train /= (batch_idx+1)
            auc_train /= (batch_idx+1)
            acc_train /= (batch_idx+1)
            if epoch%1 == 0:
                print('loss:',loss_train,'ACC:',acc_train,'AUC:',auc_train)    
                rst.append(np.array([loss_train, acc_train, auc_train]))
            early_stopping(loss_train, model)
            if early_stopping.early_stop:
                self.model = model
                print("Early stopping")
                break
        print(np.array(rst).shape)
        np.savetxt(save_folder+'train_gnn.txt', rst)

class CNN(nn.Module):
    def  __init__(self, n_task):
        super(CNN,self).__init__()
        self.conv_unit = nn.Sequential(
            # x: [b,3,94,94] => [b,32,92,92] => [b,32,46,46]
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2,stride=2,padding=0),
            # x: [b,32,46,46] => [b,64,44,44] => [b,64,22,22]            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2,stride=2,padding=0),
            # x: [b,64,22,22] => [b,64,20,20] => [b,64,10,10]        
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2,stride=2,padding=0),
            # x: [b,64,10,10] => [b,32,8,8]
            nn.Conv2d(64, 32,kernel_size=3,stride=1,padding=0)

        )
        self.fc_unit = nn.Sequential(
            nn.Linear(32*8*8,32*4*4),
            nn.ReLU(),
            nn.Dropout(0.2),     
            nn.Linear(32*4*4,120),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(120,n_task),
        )
        self.criteon = nn.CrossEntropyLoss()

    def forward(self,x):
        batchsz = x.size(0)
        # [b,3,94,94] => [b,32,8,8]
        x=self.conv_unit(x)
        x=x.view(batchsz,32*8*8)
        # [b,32*8*8] => [b,n]
        logits =self.fc_unit(x)
        # pred = F.softmax(logits,dim=1)
        # loss =self.criteon(logits, pred)
        return logits
    