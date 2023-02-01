import torch
import torch.nn as nn
import time
import argparse
import numpy as np
import apricot
import math
import pandas as pd
from utils import distance, compute_score, compute_gamma
#from cords.utils.data.dataloader.SL.adaptive.gradmatchdataloader import GradMatchDataLoader
#from cords.utils.data.dataloader.SL.models.logreg_net.py import LogisticRegNet 
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import pairwise_distances

parser = argparse.ArgumentParser()
parser.add_argument('--data', type = str, default = 'brazil')
parser.add_argument('--mode', type = int, default = 0) # 0: symmetric noise, 1: asymmetric noise
parser.add_argument('--batch_size', type = int, default = 128)
parser.add_argument('--every', type = int, default = 5)
parser.add_argument('--num_epochs', type = int, default = 50)
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

frac_list = [0.1, 0.3, 0.5, 0.7]
prob_list = [0, 0.2, 0.4, 0.6, 0.8]

mode = 'sym' if args.mode == 0 else 'asym'

df = pd.DataFrame(index=frac_list, columns=prob_list)

for frac in frac_list:
    for prob in prob_list:
        x_path = f'./data/{args.data}-train-x.npy'
        y_path = f'./data/{args.data}-train-y-{mode}-{prob}.npy' if prob != 0 else f'./data/{args.data}-train-y.npy'
        features = np.load(x_path)
        labels = np.load(y_path)
        idxs = np.random.choice(len(features), len(features), replace=False)
        features = features[idxs]
        labels = labels[idxs]
        num_classes = len(np.unique(labels))
        
        model = LogitRegression(features.shape[1], num_classes).to('cuda:0')

        dataset = MyDataset(features, labels)
        trainloader = DataLoader(dataset, batch_size = args.batch_size, shuffle = False)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=0.0003)
        start_time = time.time() 
        
        ssets = []
        weights = []
        
        for c in range(num_classes):
            ids = np.where(labels == c)[0]
            feat = features[ids]
            #print (len(feat))
            dist_mat = pairwise_distances(feat)
            dist_mat = dist_mat.max() - dist_mat
            fl = apricot.functions.facilityLocation.FacilityLocationSelection(random_state=0, metric='precomputed',
                                                                          n_samples=math.ceil(frac * len(feat)),
                                                                                              optimizer='lazy')
            sim_sub = fl.fit_transform(dist_mat)
            sset = list(np.array(np.argmax(sim_sub, axis=1)).reshape(-1))
            weight = compute_gamma(dist_mat, sset)
            #print (type(ids))
            ssets.extend(ids[sset])
            weights.extend(weight)    
            
        end_time = time.time()
        print ("CRAIG strategy data selection time is: %.4f", end_time-start_time)
        #print (len(total_greedy_list))
        #model_train
        weights = torch.Tensor(weights).to('cuda:0')
        
        start_time = time.time()
        model.train()
        
        for i in range(args.num_epochs):
            inputs, targets = torch.Tensor(features[ssets]).to('cuda:0'), torch.Tensor(labels[ssets]).long().to('cuda:0')
            outputs = model(inputs)
            loss = (weights * criterion(outputs, targets)).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        end_time = time.time()
        print ("Model training time is: %.4f", end_time-start_time)
            
        model.eval()
        test_x = np.load(f'./data/{args.data}-test-x.npy')
        test_y = np.load(f'./data/{args.data}-test-y.npy')
        test_size = len(test_y)

        pos = np.zeros(num_classes)
        Num = np.zeros(num_classes)
        pred = np.zeros(test_size)

        for i in range(test_size):
            inputs = torch.Tensor(test_x[i]).to('cuda:0')
            outputs = model(inputs)
            pred[i] = torch.argmax(outputs)
            c = int(test_y[i])
            try:
                Num[c]+=1
                if pred[i] == c:
                    pos[c]+=1
            except:
                print (c)
        result = sum(pos)/sum(Num)
        print (pos/Num)
        print (result)
                
        df.loc[frac,prob] = result        

        saved_path = f'./results/craig-{args.data}-{mode}-results.csv'

        df.to_csv(saved_path,sep=',',index=True)

            
            
    
            