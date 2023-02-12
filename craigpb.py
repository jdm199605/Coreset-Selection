import torch
import torch.nn as nn
import time
import argparse
import numpy as np
import apricot
import math
import pandas as pd
from utils import distance, compute_score, compute_gamma, CLSDataset, REGDataset, Coreset, LogitRegression, LinearRegression
from torch.utils.data import Dataset, DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--data', type = str, default = 'covtype')
parser.add_argument('--mode', type = int, default = 0) # 0: symmetric noise, 1: asymmetric noise
parser.add_argument('--batch_size', type = int, default = 128)
parser.add_argument('--num_epochs', type = int, default = 30)
parser.add_argument('--num_runs', type = int, default = 5)
parser.add_argument('--lr', type = float, default = 1e-2)
args = parser.parse_args()

frac_list = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.3, 0.5]
prob_list = [0, 0.2, 0.4, 0.6, 0.8]

mode = 'sym' if args.mode == 0 else 'asym'

df_results = pd.DataFrame(index=frac_list, columns=prob_list)
df_times = pd.DataFrame(index=frac_list, columns=prob_list)

CLS = 1 if args.data in ['covtype', 'imdbc'] else 0 #whether it is a classification problem

for frac in frac_list:
    for prob in prob_list:
        x_path = f'./data/{args.data}-train-x.npy'
        if CLS:
            y_path = f'./data/{args.data}-train-y-{mode}-{prob}.npy' if prob != 0 else f'./data/{args.data}-train-y.npy'
        else:
            y_path = f'./data/{args.data}-train-y-{prob}.npy' if prob != 0 else f'./data/{args.data}-train-y.npy'
            
        results = torch.zeros(args.num_runs)
        times = torch.zeros(args.num_runs)
        
        for run in range(args.num_runs):
            features = np.load(x_path)
            labels = np.load(y_path)
            idxs = np.random.choice(len(features), len(features), replace=False)
            features = features[idxs]
            labels = labels[idxs]

            if CLS:
                num_classes = len(np.unique(labels))
                model = LogitRegression(features.shape[1], num_classes).to('cuda:0')
            else:
                model = LinearRegression(features.shape[1]).to('cuda:0')

            dataset = CLSDataset(features, labels) if CLS else REGDataset(features, labels)
            subsetloader = DataLoader(dataset, batch_size = args.batch_size, shuffle = False)

            criterion = nn.CrossEntropyLoss() if CLS else nn.MSELoss()
            optimizer = torch.optim.SGD(model.parameters(), momentum = 0.9, lr=args.lr, weight_decay=1e-5)
            start_time = time.time() 

            ssets = []
            weights = []
            trainloader = DataLoader(dataset, batch_size = args.batch_size, shuffle=False)

            idxs = torch.arange(features.shape[0])
            #N = len(idxs)
            dist_mat = compute_score(idxs, dataset, args.batch_size)
            fl = apricot.functions.facilityLocation.FacilityLocationSelection(random_state=0, metric='precomputed',
                                                                              n_samples=math.ceil(
                                                                                  frac * features.shape[0] / args.batch_size),
                                                                              optimizer='lazy')
            sim_sub = fl.fit_transform(dist_mat)
            #print (len(sim_sub))
            temp_list = list(np.array(np.argmax(sim_sub, axis=1)).reshape(-1))
            gammas_temp = compute_gamma(dist_mat, temp_list)
            batch_wise_indices = list(trainloader.batch_sampler)

            for i in range(len(temp_list)):
                tmp = batch_wise_indices[temp_list[i]]
                ssets.extend(tmp)
                weights.extend([gammas_temp[i]] * len(tmp))

            cs_time = time.time()
            print ("CRAIG strategy data selection time is: %.4f", cs_time-start_time)

            weights = torch.Tensor(weights).to('cuda:0')
            
            assert (len(weights) == len(ssets))
            coreset = Coreset(features[ssets], labels[ssets], weights)
            #print (len(features[ssets]))
            csloader = DataLoader(coreset, batch_size = args.batch_size, shuffle = True)
            
            #start_time = time.time()
            model.train()
            
            for i in range(args.num_epochs):
                for inputs, targets, weights in csloader:
                    inputs, targets, weights = inputs.to('cuda:0'), targets.to('cuda:0'), weights.to('cuda:0')
                    outputs = model(inputs)
                    if CLS:
                        targets = targets.long()
                        loss = (weights * criterion(outputs, targets)).sum()
                    else:
                        targets = targets.unsqueeze(1)
                        #loss = (weights * criterion(outputs, targets).sum()) / sum(weights)
                        loss = (weights * criterion(outputs, targets)).sum() / sum(weights)
                        
                    #print (loss.shape)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

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
        
        r_saved_path = f'./results/craigpb-{args.data}-{mode}-results.csv' if CLS else \
                            f'./results/craigpb-{args.data}-results.csv'
        t_saved_path = f'./times/craigpb-{args.data}-{mode}-times.csv' if CLS else \
                            f'./times/craigpb-{args.data}-times.csv'
        
        df_results.to_csv(r_saved_path,sep=',',index=True)
        df_times.to_csv(t_saved_path,sep=',',index=True)
            
            
    
            