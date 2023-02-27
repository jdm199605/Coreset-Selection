import torch
import torch.nn as nn
import time
import argparse
import numpy as np
import apricot
import math
import pandas as pd
import apricot
from utils import compute_gradients, CLSDataset, REGDataset, Coreset, LogitRegression, LinearRegression, train_model, create_batch_wise_indices,MLPRegression, MLPClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import pairwise_distances
from global_variables import PATH, frac_list, prob_list

parser = argparse.ArgumentParser()
parser.add_argument('--data', type = str, default = 'imdbr')
parser.add_argument('--mode', type = int, default = 0) # 0: symmetric noise, 1: asymmetric noise
parser.add_argument('--batch_size', type = int, default = 128)
parser.add_argument('--B', type = int, default = 1024)
parser.add_argument('--every', type = int, default = 3)
parser.add_argument('--num_epochs', type = int, default = 30)
parser.add_argument('--num_runs', type = int, default = 5)
parser.add_argument('--linear', type = int, default = 0)
parser.add_argument('--num_layers', type = int, default = 3)
parser.add_argument('--num_nodes', type = int, default = 100)
parser.add_argument('--lr', type = float, default = 1e-4)
parser.add_argument('--warmup', type = int, default = 3)
parser.add_argument('--radius', type = float, default = 2.0)
parser.add_argument('--device', type = str, default = 'cuda:0')
args = parser.parse_args()

mode = 'sym' if args.mode == 0 else 'asym'

df_results = pd.DataFrame(index=frac_list, columns=prob_list)
df_times = pd.DataFrame(index=frac_list, columns=prob_list)

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
                if args.linear:
                    model = LogitRegression(features.shape[1], num_classes)
                else:
                    model = MLPClassification(features.shape[1], num_classes, args.num_layers, args.num_nodes)
            else:
                if args.linear:
                    model = LinearRegression(features.shape[1])
                else:
                    model = MLPRegression(features.shape[1], args.num_layers, args.num_nodes)

            #dataset = CLSDataset(features, labels) if CLS else REGDataset(features, labels)
            #subsetloader = DataLoader(dataset, batch_size = args.B, shuffle = False)

            criterion = nn.CrossEntropyLoss() if CLS else nn.MSELoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-5)
            budget = math.ceil(frac*len(features))
            
            start_time = time.time()
            # warm up training
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
                    
                    weights = []
                    ssets = []
                    grads = compute_gradients(model, features, labels, args.B, criterion, CLS)
                    #print (trn_gradients.shape)
                    
                    grads = grads.detach().cpu()
                    dist_mat = pairwise_distances(grads)
                    dist_mat = np.max(dist_mat) - dist_mat
                    weights_pb = np.sum(dist_mat < args.radius, axis = 1)
                    
                    if math.floor(budget / args.B) > 0:
                        fl = apricot.functions.facilityLocation.FacilityLocationSelection(random_state=0, metric='precomputed',
                                                                              n_samples=math.floor(budget / args.B),
                                                                                                  optimizer='lazy')
                        sim_sub = fl.fit_transform(dist_mat)
                        ssets_pb = list(np.array(np.argmax(sim_sub, axis=1)).reshape(-1))

                        #batch_wise_indices = list(subsetloader.batch_sampler)
                        batch_wise_indices = create_batch_wise_indices(features, arg.B)

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
                    
                    batch_size = min(len(feats), args.batch_size)
                    num_batches = int(np.ceil(len(feats)/batch_size))
        
                    model = train_model(model, criterion, optimizer, feats, labs, 1, args.batch_size, num_batches, CLS)
           
            end_time = time.time()
            print ("End-to-end time is: %.4f", end_time-start_time)

            model.eval()
            test_x = np.load(PATH + f'{args.data}-test-x.npy')
            test_y = np.load(PATH + f'{args.data}-test-y.npy')
            test_size = len(test_y)

            test_x = torch.Tensor(test_x)
            test_y = torch.Tensor(test_y)
            pred = torch.argmax(model(test_x), axis = 1)
            
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
        
        r_saved_path = f'./results/crustpb-{args.data}-{mode}-results.csv' if CLS else \
                            f'./results/crustpb-{args.data}-results.csv'
        t_saved_path = f'./times/crustpb-{args.data}-{mode}-times.csv' if CLS else \
                            f'./times/crustpb-{args.data}-times.csv'

        df_results.to_csv(r_saved_path,sep=',',index=True)
        df_times.to_csv(t_saved_path,sep=',',index=True)