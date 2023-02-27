import torch
import torch.nn as nn
import time
import argparse
import numpy as np
import apricot
import math
import pandas as pd
from utils import compute_gradients, ompwrapper, CLSDataset, REGDataset, Coreset, LogitRegression, LinearRegression, MLPRegression, MLPClassification, create_batch_wise_indices
from torch.utils.data import Dataset, DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--data', type = str, default = 'imdbr')
parser.add_argument('--mode', type = int, default = 0) # 0: symmetric noise, 1: asymmetric noise
parser.add_argument('--batch_size', type = int, default = 128)
parser.add_argument('--B', type=int, default = 2048)
parser.add_argument('--every', type = int, default = 3)
parser.add_argument('--num_epochs', type = int, default = 30)
parser.add_argument('--num_runs', type = int, default = 5)
parser.add_argument('--lr', type = float, default = 1e-4)
parser.add_argument('--warmup', type = int, default = 3)
parser.add_argument('--eps', type = float, default = 1e-4)
parser.add_argument('--linear', type = int, default = 0)
parser.add_argument('--num_layers', type = int, default = 3)
parser.add_argument('--num_nodes', type = int, default = 100)
parser.add_argument('--device', type = str, default = 'cuda:0')
parser.add_argument('--v1', type = int, default = 1)
parser.add_argument('--lam', type = float, default = 0)
args = parser.parse_args()

frac_list = [1e-4, 1e-3, 1e-2, 0.1, 0.3, 0.5]
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
        times = np.zeros(args.num_runs)
        
        for run in range(args.num_runs):
            features = np.load(x_path)
            labels = np.load(y_path)
            idxs = np.random.choice(len(features), len(features), replace=False)
            features = features[idxs]
            labels = labels[idxs]
            num_classes = len(np.unique(labels))

            if CLS:
                num_classes = len(np.unique(labels))
                if args.linear:
                    model = LogitRegression(features.shape[1], num_classes)
                else:
                    model = MLPClassification(features.shape[1], num_classes, args.num_layers, args.num_nodes)
            else:
                if args.linear:
                    model = LinearRegression(features.shape[1])
                else:
                    model = MLPRegression(features.shape[1], args.num_layers, args.num_nodes)

            criterion = nn.CrossEntropyLoss() if CLS else nn.MSELoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-5)
            budget = math.floor(frac*len(features))
            
            start_time = time.time()
            print (f'warm-up training starts. Total number of warm-up epochs: {args.warmup}.')
            if args.warmup != 0:
                batch_size = min(len(features), args.batch_size)
                num_batches = int(np.ceil(len(features)/batch_size))
                model = train_model(model, criterion, optimizer, features, labels, args.warmup, args.batch_size, num_batches, CLS)
            
            for epoch in range(args.num_epochs-args.warmup):
                print (f'Epoch {epoch + args.warmup} starts.')
                if epoch % args.every == 0:
                    cs_start = time.time()
                    print ('Time to change the Coreset')
                    grads_per_elem = compute_gradients(model, features, labels, args.B, criterion, CLS)
                    idxs = []
                    weights = []
                    trn_gradients = grads_per_elem
                    sum_val_grad = torch.sum(trn_gradients, dim = 0)
                    #ompwrapper(device, X, Y, bud, v1, lam, eps)
                    idxs_temp, weights_temp = ompwrapper(args.device, torch.transpose(trn_gradients, 0, 1), 
                                                             sum_val_grad, 
                                                             math.floor(budget / args.B), 
                                                             args.v1, args.lam, args.eps)

                    batch_wise_indices = create_batch_wise_indices(features, arg.B)
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
                        #idxs = torch.Tensor(idxs)
                        weights = torch.Tensor(weights)
                    
                    feats = features[idxs]
                    labs = labels[idxs]
                    
                    cs_end = time.time()
                    print (f'It takes {cs_end - cs_start} seconds to select a new coreset for grad-match.')
                    
                    
                    
                batch_size = min(len(feats), args.batch_size)
                num_batches = int(np.ceil(len(feats)/batch_size))
        
                model = train_model(model, criterion, optimizer, feats, labs, 1, args.batch_size, num_batches, CLS)
           
            end_time = time.time()
            print ("End-to-end time is: %.4f", end_time-start_time)

            model.eval()
            test_x = np.load(f'./data/{args.data}-test-x.npy')
            test_y = np.load(f'./data/{args.data}-test-y.npy')
            test_size = len(test_y)

            test_x = torch.Tensor(test_x)
            test_y = torch.Tensor(test_y)
            pred = torch.argmax(model(test_x), axis = 1)
            #print (pred)
            #results.append(sum(pos)/sum(Num))
            #print (pred.eq_(test_y))
            if CLS:
                pred = torch.argmax(model(test_x), axis = 1)[0]  
                results[run] = (sum(pred == test_y)/len(test_x))
            else:
                pred = model(test_x)[0]
                results[run] = math.sqrt(torch.pow(pred-test_y.unsqueeze(1), 2).sum() / len(test_y))
            times[run] = end_time - start_time
                
            print (f"frac:{frac}, prob:{prob}, run: {run}, result:{results}")
                
        df_results.loc[frac,prob] = float(results.mean())
        df_results.loc[frac,prob] = float(times.mean())
        
        r_saved_path = f'./results/gradmatch-{args.data}-{mode}-results.csv' if CLS else \
                            f'./results/gradmatch-{args.data}-results.csv'
        t_saved_path = f'./times/gradmatch-{args.data}-{mode}-times.csv' if CLS else \
                            f'./times/gradmatch-{args.data}-times.csv'

        df_results.to_csv(r_saved_path,sep=',',index=True)
        df_times.to_csv(t_saved_path,sep=',',index=True)