import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from owp_solvers import OrthogonalMP_REG_Parallel, OrthogonalMP_REG, OrthogonalMP_REG_Parallel_V1
import numpy as np
import time 
import argparse
import heapq
from torch.autograd import grad


def estimate_grads(cs_loader, model, criterion):
    model.train()
    all_grads = []
    all_targets = []
    all_preds = []
    #cs_loader = DataLoader(dataset,batch_size = 128, shuffle = False)
    for i, batch in enumerate(cs_loader):
        input, target, index = batch
        input = input.to('cuda:0')
        all_targets.append(target)
        target = target.to('cuda:0')
        
        output, feat = model(input)
        _, pred = torch.max(output, 1)
        loss = criterion(output, target).mean()
        est_grad = grad(loss, feat)
        all_grads.append(est_grad[0].detach().cpu().numpy())
        all_preds.append(pred.detach().cpu().numpy())
        
    all_grads = np.vstack(all_grads)
    #all_grads = torch.vstack(all_grads))
    all_targets = np.hstack(all_targets)
    all_preds = np.hstack(all_preds)
    return all_grads, all_targets

class FacilityLocation:
    def __init__(self, V, D=None, fnpy=None):
        if D is not None:
          self.D = D
        else:
          self.D = np.load(fnpy)

        self.D *= -1
        self.D -= self.D.min()
        self.V = V
        self.curVal = 0
        self.gains = []
        self.curr_max = np.zeros_like(self.D[0])

    def inc(self, sset, ndx):
        if len(sset + [ndx]) > 1:
            new_dists = np.stack([self.curr_max, self.D[ndx]], axis=0)
            return new_dists.max(axis=0).sum()
        else:
            return self.D[sset + [ndx]].sum()

    def add(self, sset, ndx, delta):
        self.curVal += delta
        self.gains += delta,
        self.curr_max = np.stack([self.curr_max, self.D[ndx]], axis=0).max(axis=0)
        return self.curVal

        cur_old = self.curVal
        if len(sset + [ndx]) > 1:
            self.curVal = self.D[:, sset + [ndx]].max(axis=1).sum()
        else:
            self.curVal = self.D[:, sset + [ndx]].sum()
        self.gains.extend([self.curVal - cur_old])
        return self.curVal

def _heappush_max(heap, item):
    heap.append(item)
    heapq._siftdown_max(heap, 0, len(heap)-1)


def _heappop_max(heap):
    """Maxheap version of a heappop."""
    lastelt = heap.pop()  # raises appropriate IndexError if heap is empty
    if heap:
        returnitem = heap[0]
        heap[0] = lastelt
        heapq._siftup_max(heap, 0)
        return returnitem
    return lastelt


def _heappop_max(heap):
    """Maxheap version of a heappop."""
    lastelt = heap.pop()  # raises appropriate IndexError if heap is empty
    if heap:
        returnitem = heap[0]
        heap[0] = lastelt
        heapq._siftup_max(heap, 0)
        return returnitem
    return lastelt


def lazy_greedy_heap(F, V, B):
    curVal = 0
    sset = []
    vals = []

    order = []
    heapq._heapify_max(order)

    cnt = 0
    for index in V:
      _heappush_max(order, (F.inc(sset, index), index))
      cnt += 1

    n_iter = 0
    while order and len(sset) < B:
        n_iter += 1
        if F.curVal == len(F.D):
          # all points covered
          break

        el = _heappop_max(order)
        improv = F.inc(sset, el[1])

        # check for uniques elements
        if improv > 0: 
            if not order:
                curVal = F.add(sset, el[1], improv) # NOTE: added "improv"
                sset.append(el[1])
                vals.append(curVal)
            else:
                top = _heappop_max(order)
                if improv >= top[0]:
                    curVal = F.add(sset, el[1], improv) # NOTE: added "improv"
                    sset.append(el[1])
                    vals.append(curVal)
                else:
                    _heappush_max(order, (improv, el[1]))
                _heappush_max(order, top)

    return sset, vals

def normalize(X, col):
    mu = np.mean(X[:,:col],axis=0)
    sigma = np.std(X[:,:col],axis=0)
    X[:,:col] =  (X[:,:col]-mu)/sigma
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
    
def compute_gradients(model, dataset, batch_size, criterion):
    batchloader = DataLoader(dataset, batch_size = batch_size, shuffle = False)
    
    for batch_idx, (inputs, targets) in enumerate(batchloader):
        inputs, targets = inputs.to('cuda:0'), targets.to('cuda:0')
        
        if batch_idx == 0:
            output = model(inputs)
            loss = criterion(output, targets).sum()
            l0_grads = torch.autograd.grad(loss, output)[0]
            l0_grads = l0_grads.mean(dim = 0).view(1, -1)
        else:
            output = model(inputs)
            loss = criterion(output, targets).sum()
            batch_l0_grads = torch.autograd.grad(loss, output)[0]
            batch_l0_grads = batch_l0_grads.mean(dim=0).view(1, -1)
            l0_grads = torch.cat((l0_grads, batch_l0_grads), dim=0)
            
    torch.cuda.empty_cache()
    #print (l0_grads.shape)
    return l0_grads

###################crust################################
            
        
    
    
