import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from omp_solvers import OrthogonalMP_REG_Parallel, OrthogonalMP_REG, OrthogonalMP_REG_Parallel_V1
import numpy as np
import time 
import argparse
from torch.autograd import grad

class CLSDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.Tensor(features)
        self.labels = torch.Tensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        return self.features[index], self.labels[index].long()
    
class REGDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.Tensor(features)
        self.labels = torch.Tensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        return self.features[index], self.labels[index]
    
class Coreset(Dataset):
    def __init__(self, features, labels, weights):
        self.features = torch.Tensor(features)
        self.labels = torch.Tensor(labels)
        self.weights = torch.Tensor(weights)
        
    def __getitem__(self, index):
        return self.features[index], self.labels[index], self.weights[index]
    
    def __len__(self):
        return len(self.features)
    
class LogitRegression(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(LogitRegression, self).__init__()
        self.linear = torch.nn.Linear(num_features, num_classes)
        self.num_features = num_features
    
    def forward(self, x):
        y = self.linear(x)
        return y, x
    
    def get_embed_dim(self):
        return self.num_features
    
class LinearRegression(torch.nn.Module): 
    def __init__(self, num_features):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(num_features, 1)
        self.num_features = num_features
    
    def forward(self, x):
        y =  self.linear(x)
        return y, x
    
    def get_embed_dim(self):
        return self.num_features

class MLPRegression(torch.nn.Module):
    def __init__(self, num_features, num_layers, num_nodes):
        super(MLPRegression, self).__init__()
        layers = [torch.nn.Linear(num_nodes, num_nodes) for i in range(num_layers-1)]
        layers.insert(0, nn.Linear(num_features, num_nodes))
        self.layers = torch.nn.ModuleList(layers)
        self.pred = torch.nn.Linear(num_nodes, 1)
        self.relu = torch.nn.ReLU()
        self.num_nodes = num_nodes
        
    def forward(self, x):
        for layer in self.layers:
            x = self.relu(layer(x))
        y = self.pred(x)
        return y, x
    
    def get_embed_dim(self):
        return self.num_nodes

class MLPClassification(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_layers, num_nodes):
        super(MLPClassification, self).__init__()
        layers = [torch.nn.Linear(num_nodes, num_nodes) for i in range(num_layers-1)]
        layers.insert(0, nn.Linear(num_features, num_nodes))
        self.layers = torch.nn.ModuleList(layers)
        self.relu = torch.nn.ReLU()
        self.pred = torch.nn.Linear(num_nodes, num_classes)
        self.num_nodes = num_nodes
        
    def forward(self, x):
        for layer in self.layers:
            x = self.relu(layer(x))
        y = self.pred(x)
        return y, x
    
    def get_embed_dim(self):
        return self.num_nodes
    
def train_model(model, criterion, optimizer, features, labels, num_epochs, batch_size, num_batches, CLS):
    features, labels = torch.Tensor(features), torch.Tensor(labels)
    if CLS:
        labels = labels.long()
    for epoch in range(num_epochs):
        print (f'epoch {epoch+1} starts!')
        total_loss = 0
        for b in range(num_batches):
            start = b * batch_size
            end = (b+1) * batch_size
            end = min(len(features), end)
            inputs, targets = features[start:end], labels[start:end]
            outputs = model(inputs)[0]
            if not CLS:
                targets = targets.unsqueeze(1)
            loss = criterion(outputs, targets)

            total_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print (total_loss)
    return model

def train_on_coreset_one_epoch(model, criterion, optimizer, features, labels, weights, batch_size, num_batches, CLS):
    features, labels, weights = torch.Tensor(features), torch.Tensor(labels), torch.Tensor(weights)
        #print (f'epoch {epoch+1} starts!')
    if CLS:
        labels = labels.long()
    total_loss = 0
    for b in range(num_batches):
        start = b * batch_size
        end = (b+1) * batch_size
        end = min(len(features), end)
        inputs, targets, wgts = features[start:end], labels[start:end], weights[start:end]
        outputs = model(inputs)[0]
        if not CLS:
            targets = targets.unsqueeze(1)
            loss = (wgts * criterion(outputs, targets)).sum() / wgts.sum()
        else:
            loss = (wgts * criterion(outputs, targets)).sum()

        total_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print (total_loss)
    return model
    
    
def normalize(X, col):
    mu = np.mean(X[:,:col],axis=0)
    sigma = np.std(X[:,:col],axis=0)
    X[:,:col] =  (X[:,:col]-mu)/(sigma+1e-10)
    return X

def distance(x, y, exp = 2):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    
    dist = torch.pow(x - y, exp).sum(2)
    
    return dist

def compute_score(idxs, dataset, batch_size):
    subsetloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = False, 
                                                pin_memory = True)
    
    N = 0
    g_is = []
    
    for batch_idx, (inputs, targets) in enumerate(subsetloader):
        inputs, targets = inputs, targets
        N += 1
        g_is.append(inputs.view(inputs.size()[0], -1).mean(dim=0).view(1, -1))
        
    dist_mat = torch.zeros([N, N], dtype = torch.float32)
    first_i = True
    g_is = torch.cat(g_is, dim = 0)
    dist_mat = distance(g_is, g_is).cpu()
    const = torch.max(dist_mat).item()
    dist_mat = (const - dist_mat).numpy()
    
    return dist_mat
    
def compute_gamma(dist_mat, idxs):
    gamma = [0 for i in range(len(idxs))]
    best = dist_mat[idxs]
    rep = np.argmax(best, axis = 0)
    for i in rep:
        gamma[i] += 1
    return gamma

def combine_features(features):
    X = features[0]
    
    for i in range(1, len(features)):
        X = np.concatenate((X,features[i]),axis = 0)
    
    return X
    
#####################grad_match###################
def ompwrapper(device, X, Y, bud, v1, lam, eps):
    if device == "cpu":
        reg = OrthogonalMP_REG(X.detach().numpy(), Y.detach().numpy(), nnz=bud, positive=True, lam=0)
        ind = np.nonzero(reg)[0]
        print (reg.shape, len(np.nonzero(reg)))
    else:
        if v1:
            reg = OrthogonalMP_REG_Parallel_V1(X, Y, nnz=bud,
                                             positive = True, lam = lam,
                                             tol = eps, device = device)
        else:
            reg = OrthogonalMP_REG_Parallel(X, Y, nnz=bud,
                                            positive = True, lam = lam,
                                            tol = eps, device = device)
        ind = torch.nonzero(reg).view(-1)
    return ind.tolist(), reg[ind].tolist()

def create_batch_wise_indices(features, B):
    indices = []
    batch_size = min(B, len(features))
    num_batches = int(np.ceil(len(features)/B))
    for b in range(num_batches):
        start = b * batch_size
        end = (b+1) * batch_size
        end = min(end, len(features))
        indices.append(list(range(start, end)))
    return indices      
    
#def compute_gradients(model, dataset, batch_size, criterion, CLS):
#    batchloader = DataLoader(dataset, batch_size = batch_size, shuffle = False)
#    
#    for batch_idx, (inputs, targets) in enumerate(batchloader):
#        inputs, targets = inputs.to('cuda:0'), targets.to('cuda:0')
#        
#        if batch_idx == 0:
#            output = model(inputs)
#            if not CLS:
#                targets = targets.unsqueeze(1)
#            loss = criterion(output, targets).sum()
#            l0_grads = torch.autograd.grad(loss, output)[0]
#            l0_grads = l0_grads.mean(dim = 0).view(1, -1)
#        else:
#            output = model(inputs)
#            if not CLS:
#                targets = targets.unsqueeze(1)
#            loss = criterion(output, targets).sum()
#            batch_l0_grads = torch.autograd.grad(loss, output)[0]
#            batch_l0_grads = batch_l0_grads.mean(dim=0).view(1, -1)
#            l0_grads = torch.cat((l0_grads, batch_l0_grads), dim=0)
#            
#    torch.cuda.empty_cache()
#    return l0_grads

def compute_gradients(model, features, labels, B, criterion, CLS, num_classes):
    batch_size = min(B, len(features))
    num_batches = int(np.ceil(len(features)/B))
    embDim = model.get_embed_dim()
    
    for b in range(num_batches):
        start = b * batch_size
        end =  (b+1) * batch_size
        end = min(end, len(features))
        inputs, targets = torch.Tensor(features[start:end]), torch.Tensor(labels[start:end])
        if CLS:
            targets = targets.long()
        
        if b == 0:
            output, l1 = model(inputs)
            if not CLS:
                targets = targets.unsqueeze(1)
            #print (output.shape, targets.shape)
            loss = criterion(output, targets).sum()
            l0_grads = torch.autograd.grad(loss, output)[0]
            l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
            l1_grads = l0_expand * l1.repeat(1, num_classes)                    
            l0_grads = l0_grads.mean(dim=0).view(1, -1)
            l1_grads = l1_grads.mean(dim=0).view(1, -1)      
        else:
            output, l1 = model(inputs)
            if not CLS:
                targets = targets.unsqueeze(1)
            loss = criterion(output, targets).sum()
            batch_l0_grads = torch.autograd.grad(loss, output)[0]
            batch_l0_expand = torch.repeat_interleave(batch_l0_grads, embDim, dim=1)
            #print (batch_l0_expand.shape, l1.shape)
            batch_l1_grads = batch_l0_expand * l1.repeat(1, num_classes)
            batch_l0_grads = batch_l0_grads.mean(dim=0).view(1, -1)
            batch_l1_grads = batch_l1_grads.mean(dim=0).view(1, -1)
            l0_grads = torch.cat((l0_grads, batch_l0_grads), dim=0)
            l1_grads = torch.cat((l1_grads, batch_l1_grads), dim=0)
            #print (l1_grads.shape)
            
    torch.cuda.empty_cache()
    #print (l0_grads.shape)
    #return l0_grads
    return torch.cat((l0_grads, l1_grads), dim=1)
###################prob################################
def estimate_gradients(model, features, labels, B, criterion, CLS, num_classes):
    batch_size = min(B, len(features))
    num_batches = int(np.ceil(len(features)/B))
    #embDim = model.get_embed_dim()
    
    for b in range(num_batches):
        start = b * batch_size
        end =  (b+1) * batch_size
        end = min(end, len(features))
        inputs, targets = torch.Tensor(features[start:end]), torch.Tensor(labels[start:end])
        if CLS:
            targets = targets.long()
        
        if b == 0:
            output, l1 = model(inputs)
            if not CLS:
                targets = targets.unsqueeze(1)
            #print (output.shape, targets.shape)
            loss = criterion(output, targets).sum()
            l0_grads = torch.autograd.grad(loss, l1)[0]
            #l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
            #l1_grads = l0_expand * l1.repeat(1, num_classes)                    
            l0_grads = l0_grads.mean(dim=0).view(1, -1)
            #l1_grads = l1_grads.mean(dim=0).view(1, -1)      
        else:
            output, l1 = model(inputs)
            if not CLS:
                targets = targets.unsqueeze(1)
            loss = criterion(output, targets).sum()
            batch_l0_grads = torch.autograd.grad(loss, l1)[0]
            #batch_l0_expand = torch.repeat_interleave(batch_l0_grads, embDim, dim=1)
            #print (batch_l0_expand.shape, l1.shape)
            #batch_l1_grads = batch_l0_expand * l1.repeat(1, num_classes)
            batch_l0_grads = batch_l0_grads.mean(dim=0).view(1, -1)
            #batch_l1_grads = batch_l1_grads.mean(dim=0).view(1, -1)
            l0_grads = torch.cat((l0_grads, batch_l0_grads), dim=0)
            #l1_grads = torch.cat((l1_grads, batch_l1_grads), dim=0)
            #print (l1_grads.shape)
            
    torch.cuda.empty_cache()
    #print (l0_grads.shape)
    #return l0_grads
    return l0_grads
    
            
        
    
    
