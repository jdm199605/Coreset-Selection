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
    
    def forward(self, x):
        y = self.linear(x)
        return y, y
    
class LinearRegression(torch.nn.Module): 
    def __init__(self, num_features):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(num_features, 1)
    
    def forward(self, x):
        y =  self.linear(x)
        return y, y

class MLPRegression(torch.nn.Module):
    def __init__(self, num_features, num_layers, num_nodes):
        super(MLPRegression, self).__init__()
        layers = [torch.nn.Linear(num_nodes, num_nodes) for i in range(num_layers-1)]
        layers.insert(0, nn.Linear(num_features, num_nodes))
        self.layers = torch.nn.ModuleList(layers)
        self.pred = torch.nn.Linear(num_nodes, 1)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        y = self.pred(x)
        return y, x

class MLPClassification(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_layers, num_nodes):
        super(MLPRegression, self).__init__()
        layers = [torch.nn.Linear(num_nodes, num_nodes) for i in range(num_layers-1)]
        layers.insert(0, nn.Linear(num_features, num_nodes))
        self.layers = torch.nn.ModuleList(layers)
        self.pred = torch.nn.Linear(num_nodes, num_classes)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        y = self.pred(x)
        return y, x
    
def train_model(model, criterion, optimizer, features, labels, num_epochs, batch_size, num_batches, CLS):
    for epoch in range(num_epochs):
        print (f'epoch {epoch+1} starts!')
        total_loss = 0
        for b in range(num_batches):
            start = b * batch_size
            end = (b+1) * batch_size
            end = min(len(features), end)
            inputs, targets = torch.Tensor(features[start:end]), torch.Tensor(labels[start:end])
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

def compute_gradients(model, features, labels, B, criterion, CLS):
    batch_size = min(B, len(features))
    num_batches = int(np.ceil(len(features)/B))
    
    for b in range(num_batches):
        start = b * batch_size
        end =  (b+1) * batch_size
        end = min(end, len(features))
        inputs, targets = torch.Tensor(features[start:end]), torch.Tensor(labels[start:end])
        
        if b == 0:
            output, last = model(inputs)
            if not CLS:
                targets = targets.unsqueeze(1)
            #print (output.shape, targets.shape)
            loss = criterion(output, targets).sum()
            l0_grads = torch.autograd.grad(loss, output)[0]
            l0_grads = l0_grads.mean(dim = 0).view(1, -1)
        else:
            output, last = model(inputs)
            if not CLS:
                targets = targets.unsqueeze(1)
            loss = criterion(output, targets).sum()
            batch_l0_grads = torch.autograd.grad(loss, output)[0]
            batch_l0_grads = batch_l0_grads.mean(dim=0).view(1, -1)
            l0_grads = torch.cat((l0_grads, batch_l0_grads), dim=0)
            
    torch.cuda.empty_cache()
    return l0_grads
###################crust################################
            
        
    
    
