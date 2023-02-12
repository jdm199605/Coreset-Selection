import torch
import torch.nn as nn
import time
import argparse
import numpy as np
import math
import pandas as pd
from utils import CLSDataset, REGDataset, LogitRegression, LinearRegression
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import pairwise_distances

parser = argparse.ArgumentParser()
parser.add_argument('--data', type = str, default = 'covtype')
parser.add_argument('--mode', type = int, default = 0) # 0: symmetric noise, 1: asymmetric noise
parser.add_argument('--batch_size', type = int, default = 128)
parser.add_argument('--num_epochs', type = int, default = 30)
parser.add_argument('--method', type = str, default = 'full')
parser.add_argument('--num_runs', type = int, default = 5)
parser.add_argument('--lr', type = float, default = 1e-2)
parser.add_argument('--device', type = str, default = 'cuda:0')
args = parser.parse_args()

frac_list = [1] if args.method == 'full' else [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.3, 0.5]
prob_list = [0, 0.2, 0.4, 0.6, 0.8]

mode = 'sym' if args.mode == 0 else 'asym'

df_results = pd.DataFrame(index=frac_list, columns=prob_list)
df_times= pd.DataFrame(index=frac_list, columns=prob_list)

CLS = 1 if args.data in ['covtype', 'imdbc'] else 0 #whether it is a classification problem

for frac in frac_list:
    for prob in prob_list:
        x_path = f'./data/{args.data}-train-x.npy'
        if CLS:
            y_clean_path = f'./data/{args.data}-train-y.npy'
            y_path = f'./data/{args.data}-train-y-{mode}-{prob}.npy' if prob != 0 else y_clean_path
        else:
            y_clean_path = f'./data/{args.data}-train-y.npy'
            y_path = f'./data/{args.data}-train-y-{prob}.npy' if prob != 0 else y_clean_path
        #print (x_path, y_path)
        
        results = torch.zeros(args.num_runs)
        times = np.zeros(args.num_runs)
        
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
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum = 0.9, weight_decay = 1e-5)
            start_time = time.time() 

            model.train()

            for epoch in range(args.num_epochs):
                print (f'epoch {epoch+1} starts!')
                total_loss = 0
                for inputs, targets in trainloader:
                    inputs, targets = inputs.to('cuda:0'), targets.to('cuda:0')
                    #print (inputs[0])
                    outputs = model(inputs)
                    if not CLS:
                        targets = targets.unsqueeze(1)
                    loss = criterion(outputs, targets)
                    
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
            times[run] = end_time - start_time
            print (f"frac:{frac}, prob:{prob}, run: {run}, result:{results}")

        df_results.loc[frac,prob] = float(results.mean())
        df_times.loc[frac,prob] = float(times.mean())

        r_saved_path = f'./results/{args.method}-{args.data}-{mode}-results.csv' if CLS else \
                            f'./results/{args.method}-{args.data}-results.csv'
        t_saved_path = f'./times/{args.method}-{args.data}-{mode}-times.csv' if CLS else \
                            f'./times/{args.method}-{args.data}-times.csv'

        df_results.to_csv(r_saved_path,sep=',',index=True)
        df_times.to_csv(t_saved_path, sep = ',',index=True)
        

            
            
    
            