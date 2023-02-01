import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time 
import argparse
from sklearn.metrics import pairwise_distances
from utils import estimate_grads, FacilityLocation, lazy_greedy_heap
import time

parser = argparse.ArgumentParser()
parser.add_argument('--t',type=float,default=2.0)
parser.add_argument('--num_runs',type=int,default=1)
parser.add_argument('--mode',type=int,default=0)

mode = parser.parse_args().mode

args = parser.parse_args()
                    
input_dim = 54
num_classes = 7
num_epochs = 60

frac_list = [0.1, 0.3, 0.5, 0.7]
prob_list = [0, 0.2, 0.4, 0.6, 0.8]

df = pd.DataFrame(index=frac_list, columns=prob_list)

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.Tensor(data)
        self.labels = torch.Tensor(labels).long()
        self.wholedata = self.data
        self.wholelabels = self.labels
        
    def __len__(self):
        return len(self.data)
  
    def __getitem__(self, index):
        try:
            return self.data[index], self.labels[index], index
        except:
            print (print (index, len(self.data)))
            
    def switch_data(self):
        self.data = self.wholedata
        self.labels = self.wholelabels
    
    def adjust(self,idx):
        self.data = self.wholedata[idx, ...]
        self.labels = self.wholelabels[idx,...]
        
    def fetch(self,targets):
        wholelabels_np = np.array(self.wholelabels)
        uniq_labels = np.unique(wholelabels_np)
        idx_dict = {}
        for label in uniq_labels:
            idx_dict[label] = np.where(wholelabels_np == label)[0]

        idx_list = []
        for target in targets:
            idx_list.append(np.random.choice(idx_dict[target.item()], 1))
        idx_list = np.array(idx_list).flatten()
        train_data = []
        for idx in idx_list:
            data = self.wholedata[idx]
            train_data.append(data[None, ...])
        train_data = torch.cat(train_data, dim=0)
        return train_data

train_x = np.load('./data/covtype-train-x.npy')

test_x = np.load('./data/covtype-test-x.npy')
test_y = np.load('./data/covtype-test-y.npy')

class logistic(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(logistic, self).__init__()
        self.linear1 = nn.Linear(input_dim, 54)
        self.linear2 = nn.Linear(54, num_classes)

    def forward(self, x):
        feats = self.linear1(x)
        out = self.linear2(feats)
        return out, feats
    
model = logistic(input_dim, num_classes).to('cuda:0')

for p in prob:
    if p == 0:
        path = './data/train-y-0.1.npy'
    else: 
        if mode == 0:
            pass
            #path = f'./data/train-y-0.1-asym-{p}.npy'
        else:
            pass
            #path = f'./data/train-y-0.1-sym-{p}.npy'
    path = f'./data/covtype-train-y-sym-{p}.npy'  ##################
    train_y = np.load(path)
    
    for fl_ratio in fl_ratios:
        result = []
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0001,weight_decay=0.0003)
        
        dataset = MyDataset(train_x, train_y)
        dataloader = DataLoader(dataset,batch_size = 128, shuffle = False)
        cs_loader = DataLoader(dataset,batch_size = 128, shuffle = False)
        print ('ratio:{},prob:{}'.format(fl_ratio,p))
        #for run in range(num_runs):
        cur = time.time()
        for epoch in range(num_epochs):
          #calculate the gradient
            if epoch >= 5:
                dataset.switch_data()
                grads_all, labels = estimate_grads(cs_loader, model, criterion)
                print (grads_all.shape)
          #per-class clustering
                ssets = []
                weights = []

                for c in range(num_classes):
                    sample_ids = np.where((labels==c)==True)[0]
                    grads = grads_all[sample_ids]
                    start_2 = time.time() ####################threshold
                    dists = pairwise_distances(grads)
                    print (f'time:{time.time()-start_2}')
                    weight = np.sum(dists < args.t, axis=1) ####################threshold
                    V = range(len(grads))
                    F = FacilityLocation(V, D=dists)
                    B = int(fl_ratio * len(grads))
                    sset, vals = lazy_greedy_heap(F, V, B)
                    weights.extend(weight[sset].tolist())
                    sset = sample_ids[np.array(sset)]
                    ssets += list(sset)

                weights = torch.FloatTensor(weights)
                dataset.adjust(ssets)

                model.train()

                for i, batch in enumerate(dataloader):
                    input, target, index = batch

                    #input_b = dataloader.dataset.fetch(target)
                    #lam = np.random.beta(1, 0.1)
                    #input = lam * input + (1 - lam) * input_b
                    c_weights = weights[index]
                    c_weights = c_weights.type(torch.FloatTensor)
                    c_weights =  c_weights / c_weights.sum()
                    c_weights = c_weights.to('cuda:0')

                    # measure data loading time

                    input = input.type(torch.FloatTensor).to('cuda:0')
                    target = target.to('cuda:0')

                    # compute output
                    output, feats = model(input)
                    loss = criterion(output, target)
                    loss = (loss * c_weights).sum()

                    # compute gradient and do SGD step
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            else:
              # Forward pass
                for inputs, targets, index in dataloader:
                    #print (inputs.shape)
                    inputs = inputs.to('cuda:0')
                    targets = targets.to('cuda:0')
                    outputs, feats = model(inputs)
                    loss = criterion(outputs, targets)

              # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        print (time.time()-cur)

        model.eval()
        test_size = len(test_y)
        test_size = len(test_y)

        pos = np.zeros(num_classes)
        Num = np.zeros(num_classes)
        pred = np.zeros(test_size)

        for i in range(test_size):
            inputs = torch.Tensor(test_x[i]).to('cuda:0')
            outputs = model(inputs)
            pred[i] = torch.argmax(outputs[0].data)
            c = int(test_y[i])
            try:
                Num[c]+=1
                if pred[i] == c:
                    pos[c]+=1
            except:
                print (c)
        result. = sum(pos)/sum(Num)
                
        df.loc[frac,prob] = result


saved_path = './results/crust-{arg.data}-{args.mode}-results.csv'

df.to_csv(saved_path,sep=',',index=True)

        
