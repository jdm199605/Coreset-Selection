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
    
    def forward(self, inputs):
        output = self.linear(inputs)
        return output
    
class LinearRegression(torch.nn.Module): 
    def __init__(self, num_features):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(num_features, 1)
    
    def forward(self, inputs):
        return self.linear(inputs)

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
        reg = OrthogonalMP_REG(X.numpy(), Y.numpy(), nnz=bud, positive=True, lam=0)
        ind = np.nonzero(reg)[0]
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
    
def compute_gradients(model, dataset, batch_size, criterion, CLS):
    batchloader = DataLoader(dataset, batch_size = batch_size, shuffle = False)
    
    for batch_idx, (inputs, targets) in enumerate(batchloader):
        inputs, targets = inputs.to('cuda:0'), targets.to('cuda:0')
        
        if batch_idx == 0:
            output = model(inputs)
            if not CLS:
                targets = targets.unsqueeze(1)
            loss = criterion(output, targets).sum()
            l0_grads = torch.autograd.grad(loss, output)[0]
            l0_grads = l0_grads.mean(dim = 0).view(1, -1)
        else:
            output = model(inputs)
            if not CLS:
                targets = targets.unsqueeze(1)
            loss = criterion(output, targets).sum()
            batch_l0_grads = torch.autograd.grad(loss, output)[0]
            batch_l0_grads = batch_l0_grads.mean(dim=0).view(1, -1)
            l0_grads = torch.cat((l0_grads, batch_l0_grads), dim=0)
            
    torch.cuda.empty_cache()
    return l0_grads

###################crust################################
            
        
    
    
