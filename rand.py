import torch
import torch.nn as nn
import time
import argparse
import numpy as np
import math
import pandas as pd
from utils import CLSDataset, REGDataset, LogitRegression, LinearRegression, MLPRegression, MLPClassification, train_model
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import pairwise_distances
from global_variables import PATH, frac_list, prob_list

parser = argparse.ArgumentParser()
parser.add_argument('--data', type = str, default = 'imdbr')
parser.add_argument('--mode', type = int, default = 0) # 0: symmetric noise, 1: asymmetric noise
parser.add_argument('--batch_size', type = int, default = 128)
parser.add_argument('--num_epochs', type = int, default = 30)
parser.add_argument('--method', type = str, default = 'full')
parser.add_argument('--num_runs', type = int, default = 1)
parser.add_argument('--lr', type = float, default = 1e-5)
parser.add_argument('--device', type = str, default = 'cuda:0')
parser.add_argument('--linear', type = int, default = 0)
parser.add_argument('--num_layers', type = int, default = 3)
parser.add_argument('--num_nodes', type = int, default = 100)
parser.add_argument('--version', type=int, default = 0)

args = parser.parse_args()

frac_list = [1] if args.method == 'full' else frac_list

mode = 'sym' if args.mode == 0 else 'asym'

df_results = pd.DataFrame(index=frac_list, columns=prob_list)
df_times= pd.DataFrame(index=frac_list, columns=prob_list)

CLS = 1 if args.data in ['covtype', 'imdbc'] else 0 #whether it is a classification problem

for frac in frac_list:
    for prob in prob_list:
        x_path = PATH + f'{args.data}-train-x.npy'
        if CLS:
            y_clean_path = PATH + f'{args.data}-train-y.npy'
            y_path = PATH + f'{args.data}-train-y-{mode}-{prob}.npy' if prob != 0 else y_clean_path
        else:
            y_clean_path = PATH + f'{args.data}-train-y.npy'
            y_path = PATH + f'{args.data}-train-y-{prob}.npy' if prob != 0 else y_clean_path
        #print (x_path, y_path)
        
        results = torch.zeros(args.num_runs)
        times = np.zeros(args.num_runs)
        
        for run in range(args.num_runs):
            features = np.load(x_path)
            labels = np.load(y_path)
            idxs = np.random.choice(len(features), int(frac*len(features)), replace=False)
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

            features = features[idxs]
            labels = labels[idxs]

            #dataset = CLSDataset(features, labels) if CLS else REGDataset(features, labels)
            #trainloader = DataLoader(dataset, batch_size = args.batch_size, shuffle = True)

            criterion = nn.CrossEntropyLoss() if CLS else nn.MSELoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay = 1e-5)
            start_time = time.time() 

            model.train()

            batch_size = min(len(features), args.batch_size)
            num_batches = int(np.ceil(len(features)/batch_size))
            
            model = train_model(model, criterion, optimizer, features, labels, args.num_epochs, batch_size, num_batches, CLS)

            end_time = time.time()
            print ("Model training time is: %.4f", end_time-start_time)

            model.eval()
            test_x = np.load(PATH + f'{args.data}-test-x.npy')
            test_y = np.load(PATH + f'{args.data}-test-y.npy')
            test_size = len(test_y)

            test_x = torch.Tensor(test_x)
            test_y = torch.Tensor(test_y)
            if CLS:
                pred = torch.argmax(model(test_x), axis = 1)[0]
                results[run] = (sum(pred == test_y)/len(test_x))
            else:
                pred = model(test_x)[0]
                results[run] = math.sqrt(torch.pow(pred-test_y.unsqueeze(1), 2).sum() / len(test_y))
            times[run] = end_time - start_time
            print (f"frac:{frac}, prob:{prob}, run: {run}, result:{results}")

        df_results.loc[frac,prob] = float(results.mean())
        df_times.loc[frac,prob] = float(times.mean())

        if args.version == 0:
            r_saved_path = f'./results/gradmatch-{args.data}-{mode}-results.csv' if CLS else \
                                f'./results/gradmatch-{args.data}-results.csv'
            t_saved_path = f'./times/gradmatch-{args.data}-{mode}-times.csv' if CLS else \
                                f'./times/gradmatch-{args.data}-times.csv'
        else:
            r_saved_path = f'./results/gradmatch-{args.data}-{mode}-results-{args.version}.csv' if CLS else \
                                f'./results/gradmatch-{args.data}-results-{args.version}.csv'
            t_saved_path = f'./times/gradmatch-{args.data}-{mode}-times-{args.version}.csv' if CLS else \
                                f'./times/gradmatch-{args.data}-times-{args.version}.csv'

        df_results.to_csv(r_saved_path,mode = 'a', sep=',',index=True)
        df_times.to_csv(t_saved_path, mode = 'a', sep = ',',index=True)
        

            
            
    
            