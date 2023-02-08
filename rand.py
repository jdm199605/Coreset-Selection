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
parser.add_argument('--data', type = str, default = 'taxi')
parser.add_argument('--mode', type = int, default = 0) # 0: symmetric noise, 1: asymmetric noise
parser.add_argument('--batch_size', type = int, default = 256)
parser.add_argument('--num_epochs', type = int, default = 50)
parser.add_argument('--method', type = str, default = 'full')
parser.add_argument('--num_runs', type = int, default = 5)
parser.add_argument('--lr', type = float, default = 1e-4)
parser.add_argument('--device', type = str, default = 'cuda:0')
args = parser.parse_args()

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

frac_list = [1] if args.method == 'full' else [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.3, 0.5]
prob_list = [0, 0.2, 0.4, 0.6, 0.8]

mode = 'sym' if args.mode == 0 else 'asym'

df = pd.DataFrame(index=frac_list, columns=prob_list)

CLS = 1 if args.data in ['covtype', 'imdbc'] else 0 #whether it is a classification problem

for frac in frac_list:
    for prob in prob_list:
        x_path = f'./data/{args.data}-train-x.npy'
        if CLS:
            y_path = f'./data/{args.data}-train-y-{mode}-{prob}.npy' if prob != 0 else f'./data/{args.data}-train-y.npy'
        else:
            y_path = f'./data/{args.data}-train-y-{prob}.npy' if prob != 0 else f'./data/{args.data}-train-y.npy'
        #print (x_path, y_path)
        results = torch.zeros(args.num_runs)
        for run in range(args.num_runs):
            features = np.load(x_path)
            labels = np.load(y_path)
            idxs = np.random.choice(len(features), int(frac*len(features)), replace=False)
            if CLS:
                num_classes = len(np.unique(labels))
                model = LogitRegression(features.shape[1], num_classes).to('cuda:0')
            else:
                model = LinearRegression(features.shape[1]).to('cuda:0')

            features = features[idxs]
            labels = labels[idxs]

            dataset = CLSDataset(features, labels) if CLS else REGDataset(features, labels)
            trainloader = DataLoader(dataset, batch_size = args.batch_size, shuffle = True)

            criterion = nn.CrossEntropyLoss() if CLS else nn.MSELoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay = 0.003)
            start_time = time.time() 

            model.train()

            for epoch in range(args.num_epochs):
                print (f'epoch {epoch+1} starts!')
                total_loss = 0
                for inputs, targets in trainloader:
                    inputs, targets = inputs.to('cuda:0'), targets.to('cuda:0')
                    #print (inputs[0])
                    outputs = model(inputs)
                    #print (outputs.shape, targets.shape)
                    loss = criterion(outputs, targets.unsqueeze(1))
                    total_loss += loss
            
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                print (total_loss)

            end_time = time.time()
            print ("Model training time is: %.4f", end_time-start_time)

            model.eval()
            test_x = np.load(f'./data/{args.data}-test-x.npy')
            test_y = np.load(f'./data/{args.data}-test-y.npy')
            test_size = len(test_y)

            test_x = torch.Tensor(test_x).to('cuda:0')
            test_y = torch.Tensor(test_y).to('cuda:0')
            if CLS:
                pred = torch.argmax(model(test_x), axis = 1)  
                results[run] = (sum(pred == test_y)/len(test_x))
            else:
                pred = model(test_x)
                results[run] = math.sqrt(torch.pow(pred-test_y.unsqueeze(1), 2).sum() / len(test_y))
            print (f"frac:{frac}, prob:{prob}, run: {run}, result:{results}")

        df.loc[frac,prob] = float(results.mean())

        saved_path = f'./results/{args.method}-{args.data}-{mode}-results.csv' if CLS else \
                            f'./results/{args.method}-{args.data}-results.csv'

        df.to_csv(saved_path,sep=',',index=True)

            
            
    
            