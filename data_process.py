import pandas as pd
import numpy as np
import argparse
from utils import normalize

parser = argparse.ArgumentParser()
parser.add_argument('--data',type=str, default='brazil')
parser.add_argument('--r',type=float, default = 0.7)

args = parser.parse_args()

if args.data == 'covtype':
    path = './data/covtype.csv'
    df = pd.read_table(path,sep=',')
    dataset = df.to_numpy().astype('float64')
else:
    path = f'./data/{args.data}.npy'
    dataset = np.load(path)

num_features = dataset.shape[1]-1
size = dataset.shape[0]
#dataset = normalize(dataset, num_features)

#split the data
train_index = np.random.choice(size, int(args.r*size),replace=False)
test_index = list(set(range(size)) - set(train_index))
    
train_x = dataset[train_index, :-1]
train_y = dataset[train_index, -1]

test_x = dataset[test_index, :-1]
test_y = dataset[test_index, -1]

if args.data == 'covtype':
    train_y -= 1
    test_y -= 1

train_x_path = f'./data/{args.data}-train-x.npy'
train_y_path = f'./data/{args.data}-train-y.npy'

test_x_path = f'./data/{args.data}-test-x.npy'
test_y_path = f'./data/{args.data}-test-y.npy'

np.save(train_x_path, train_x)
np.save(train_y_path, train_y)
np.save(test_x_path, test_x)
np.save(test_y_path, test_y)



