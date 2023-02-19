import torch
import torch.nn as nn
import time
import argparse
import numpy as np
import apricot
import math
import pandas as pd
import apricot
from utils import compute_gradients, CLSDataset, REGDataset, Coreset, LogitRegression, LinearRegression
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import pairwise_distances

parser = argparse.ArgumentParser()
parser.add_argument('--data', type = str, default = 'imdbr')
parser.add_argument('--mode', type = int, default = 0) # 0: symmetric noise, 1: asymmetric noise
parser.add_argument('--batch_size', type = int, default = 128)
parser.add_argument('--B', type = int, default = 1024)
parser.add_argument('--every', type = int, default = 3)
parser.add_argument('--num_epochs', type = int, default = 30)
parser.add_argument('--num_runs', type = int, default = 5)
parser.add_argument('--lr', type = float, default = 1e-4)
parser.add_argument('--warmup', type = int, default = 3)
parser.add_argument('--radius', type = float, default = 2.0)
parser.add_argument('--device', type = str, default = 'cuda:0')
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
            num_classes = len(np.unique(labels))

            if CLS:
                num_classes = len(np.unique(labels))
                model = LogitRegression(features.shape[1], num_classes).to('cuda:0')
            else:
                model = LinearRegression(features.shape[1]).to('cuda:0')

            dataset = CLSDataset(features, labels) if CLS else REGDataset(features, labels)
            subsetloader = DataLoader(dataset, batch_size = args.B, shuffle = False)

            criterion = nn.CrossEntropyLoss() if CLS else nn.MSELoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-5)
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
                        if not CLS:
                            targets = targets.unsqueeze(1)
                        loss = criterion(outputs, targets)
                        
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
            
            for epoch in range(args.num_epochs-args.warmup):
                print (f'Epoch {epoch + args.warmup} starts.')
                if epoch % args.every == 0:
                    cs_start = time.time()
                    print ('Time to change the Coreset')
                    
                    weights = []
                    ssets = []
                    grads = compute_gradients(model, dataset, args.B, criterion, CLS)
                    #print (trn_gradients.shape)
                    
                    grads = grads.detach().cpu()
                    dist_mat = pairwise_distances(grads)
                    dist_mat = np.max(dist_mat) - dist_mat
                    weights_pb = np.sum(dist_mat < args.radius, axis = 1)
                    
                    fl = apricot.functions.facilityLocation.FacilityLocationSelection(random_state=0, metric='precomputed',
                                                                          n_samples=math.ceil(budget / args.B),
                                                                                              optimizer='lazy')
                    sim_sub = fl.fit_transform(dist_mat)
                    ssets_pb = list(np.array(np.argmax(sim_sub, axis=1)).reshape(-1))
                    #gammas_temp = compute_gamma(dist_mat, temp_list)
                    batch_wise_indices = list(subsetloader.batch_sampler)

                    for i in range(len(ssets_pb)):
                        tmp = batch_wise_indices[ssets_pb[i]]
                        ssets.extend(tmp)
                        weights.extend([weights_pb[i]] * len(tmp))
                    
                    remain = budget - len(ssets)
                    
                    if remain > 0:
                        remain_list = set(np.arange(len(features))).difference(set(ssets))
                        new_idxs = np.random.choice(list(remain_list), remain, replace = False)
                        ssets.extend(new_idxs)
                        weights.extend([1] * remain)
                        weights = torch.Tensor(weights)
                    
                    feats = features[ssets]
                    labs = labels[ssets]
                    
                    cs_end = time.time()
                    print (f'It takes {cs_end - cs_start} seconds to select a new coreset for crust.')
                    
                    coreset = Coreset(feats, labs, weights)
                    trainloader = DataLoader(coreset, batch_size = args.batch_size, shuffle = True)
                    
                for inputs, targets, weights in trainloader:
                    inputs, targets, weights = torch.Tensor(inputs).to('cuda:0'), torch.Tensor(targets).to('cuda:0'), torch.Tensor(weights).to('cuda:0')
                    outputs = model(inputs)
                    if CLS:
                        targets = targets.long()
                        loss = (weights * criterion(outputs, targets)).sum()
                    else:
                        targets = targets.unsqueeze(1)
                        loss = (weights * criterion(outputs, targets)).sum() / sum(weights)
                    
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
        
        r_saved_path = f'./results/crustpb-{args.data}-{mode}-results.csv' if CLS else \
                            f'./results/crustpb-{args.data}-results.csv'
        t_saved_path = f'./times/crustpb-{args.data}-{mode}-times.csv' if CLS else \
                            f'./times/crustpb-{args.data}-times.csv'

        df_results.to_csv(r_saved_path,sep=',',index=True)
        df_times.to_csv(t_saved_path,sep=',',index=True)