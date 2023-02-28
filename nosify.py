import numpy as np
import argparse
from global_variables import PATH

parser = argparse.ArgumentParser()
parser.add_argument('--data', type = str, default = 'imdbr')
parser.add_argument('--r', type = float, default = 5)
parser.add_argument('--pos', type = int, default = 1)
data = parser.parse_args().data
r = parser.parse_args().r
pos = parser.parse_args().pos

probs = [0.2, 0.4, 0.6, 0.8]

path = f'./data/{data}-train-y.npy'
train_y = np.load(path)

uniq_y = np.unique(train_y)
num_classes = len(uniq_y)

if data in ['imdbc','covtype']:
    for prob in probs:
        Y = train_y.copy()
        for i in range(len(train_y)):
            if np.random.random() < prob:
                Y[i] = (train_y[i]+1) % num_classes

        path = PATH + f'{data}-train-y-asym-{prob}.npy'
        np.save(path, Y)

        Y = train_y.copy()
        for i in range(len(Y)):
            if np.random.random() < prob:
                new_y = np.random.randint(num_classes)
                while new_y == train_y[i]:
                    new_y = np.random.randint(num_classes)
                Y[i] = new_y

        path = PATH + f'{data}-train-y-sym-{prob}.npy'
        np.save(path, Y)
else:
    for prob in probs:
        Y = train_y.copy()
        bound = max(Y)
        noises = np.random.binomial(1, prob, (len(Y),)) * np.random.uniform(0, r*bound, (len(Y),)) if pos else np.random.binomial(1, prob, (len(Y),)) * np.random.uniform(-r*bound, r*bound, (len(Y),))
        Y += noises
        
        path = PATH + f'{data}-train-y-{prob}'
        np.save(path, Y)
        