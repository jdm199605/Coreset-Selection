import torch
import torch.nn as nn
import time
import argparse
import numpy as np
import apricot
import math
import pandas as pd
from utils import compute_gradients, ompwrapper
from torch.utils.data import Dataset, DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--data', type = str, default = 'covtype')
parser.add_argument('--mode', type = int, default = 0) # 0: symmetric noise, 1: asymmetric noise
parser.add_argument('--batch_size', type = int, default = 128)
parser.add_argument('--every', type = int, default = 5)
parser.add_argument('--num_epochs', type = int, default = 50)
parser.add_argument('--num_runs', type = int, default = 5)
parser.add_argument('--lr', type = float, default = 1e-2)
parser.add_argument('--warmup', type = int, default = 5)
parser.add_argument('--eps', type = float, default = 1e-4)
parser.add_argument('--device', type = str, default = 'cuda:0')
parser.add_argument('--v1', type = int, default = 1)
parser.add_argument('--lam', type = float, default = 0)
args = parser.parse_args()

class MyDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.Tensor(features)
        self.labels = torch.Tensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        return self.features[index], self.labels[index].long()

class LogitRegression(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(LogitRegression, self).__init__()
        self.linear = torch.nn.Linear(num_features, num_classes)
    
    def forward(self, inputs):
        output = self.linear(inputs)
        return output

class Coreset(Dataset):
    def __init__(self, features, labels, weights):
        self.features = torch.Tensor(features)
        self.labels = torch.Tensor(labels)
        self.weights = torch.Tensor(weights)
        
    def __getitem__(self, index):
        return self.features[index], self.labels[index].long(), self.weights[index]
    
    def __len__(self):
        return len(self.features)

frac_list = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.3, 0.5]
prob_list = [0, 0.2, 0.4, 0.6, 0.8]

mode = 'sym' if args.mode == 0 else 'asym'

df = pd.DataFrame(index=frac_list, columns=prob_list)

for frac in frac_list:
    for prob in prob_list:
        x_path = f'./data/{args.data}-train-x.npy'
        y_path = f'./data/{args.data}-train-y-{mode}-{prob}.npy' if prob != 0 else f'./data/{args.data}-train-y.npy'
        results = torch.zeros(5)
        
        for run in range(args.num_runs):
            features = np.load(x_path)
            labels = np.load(y_path)
            idxs = np.random.choice(len(features), len(features), replace=False)
            features = features[idxs]
            labels = labels[idxs]
            num_classes = len(np.unique(labels))

            model = LogitRegression(features.shape[1], num_classes).to('cuda:0')

            dataset = MyDataset(features, labels)
            subsetloader = DataLoader(dataset, batch_size = args.batch_size, shuffle = False)

            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-8)
            budget = math.ceil(frac*len(features))
            
            start_time = time.time()
            # warm up training
            print (f'warm-up training starts. Total number of warm-up epochs: {args.warmup}.')
            if args.warmup != 0:
                warmuploader = DataLoader(dataset, batch_size = args.batch_size, shuffle = False)
                for epoch in range(args.warmup):
                    for inputs, targets in warmuploader:
                        inputs, targets = inputs.to('cuda:0'), targets.to('cuda:0')
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
            
            for epoch in range(args.num_epochs-args.warmup):
                print (f'Epoch {epoch + args.warmup} starts.')
                if epoch % args.every == 0:
                    cs_start = time.time()
                    print ('Time to change the Coreset')
                    grads_per_elem = compute_gradients(model, dataset, args.batch_size, criterion)
                    idxs = []
                    weights = []
                    trn_gradients = grads_per_elem
                    sum_val_grad = torch.sum(trn_gradients, dim = 0)
                    #ompwrapper(device, X, Y, bud, v1, lam, eps)
                    idxs_temp, weights_temp = ompwrapper(args.device, torch.transpose(trn_gradients, 0, 1), 
                                                             sum_val_grad, 
                                                             math.ceil(budget / args.batch_size), 
                                                             args.v1, args.lam, args.eps)

                    batch_wise_indices = list(subsetloader.batch_sampler)
                    for i in range(len(idxs_temp)):
                        tmp = batch_wise_indices[idxs_temp[i]]
                        idxs.extend(tmp)
                        weights.extend(list(weights_temp[i] * np.ones(len(tmp))))
                    
                    remain = budget - len(idxs)
                    
                    if remain > 0:
                        remain_list = set(np.arange(len(features))).difference(set(idxs))
                        new_idxs = np.random.choice(list(remain_list), remain, replace = False)
                        idxs.extend(new_idxs)
                        weights.extend([1] * remain)
                        idxs = torch.Tensor(idxs)
                        weights = torch.Tensor(weights)
                    
                    feats = features[idxs]
                    labs = labels[idxs]
                    
                    cs_end = time.time()
                    print (f'It takes {cs_end - cs_start} seconds to select a new coreset for grad-match.')
                    
                    coreset = Coreset(feats, labs, weights)
                    trainloader = DataLoader(coreset, batch_size = args.batch_size, shuffle = True)
                    
                for inputs, targets, weights in trainloader:
                    inputs, targets, weights = torch.Tensor(inputs).to('cuda:0'), torch.Tensor(targets).to('cuda:0'), torch.Tensor(weights).to('cuda:0')
                    output = model(inputs)
                    loss = (weights * criterion(output, targets)).sum()
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
           
            end_time = time.time()
            print ("End-to-end time is: %.4f", end_time-start_time)

            model.eval()
            test_x = np.load(f'./data/{args.data}-test-x.npy')
            test_y = np.load(f'./data/{args.data}-test-y.npy')
            test_size = len(test_y)

            test_x = torch.Tensor(test_x).to('cuda:0')
            test_y = torch.Tensor(test_y).to('cuda:0')
            pred = torch.argmax(model(test_x), axis = 1)
            #print (pred)
            #results.append(sum(pos)/sum(Num))
            #print (pred.eq_(test_y))
            results[run] = (sum(pred == test_y)/len(test_x))
            print (f"frac:{frac}, prob:{prob}, run: {run}, result:{results}")
                
        df.loc[frac,prob] = results.mean()
        
        saved_path = f'./results/gradmatch-{args.data}-{mode}-results.csv'

        df.to_csv(saved_path,sep=',',index=True)