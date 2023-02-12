import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', type = str, default = 'taxi')
data = parser.parse_args().data

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

        path = f'./data/{data}-train-y-asym-{prob}.npy'
        np.save(path, Y)

        Y = train_y.copy()
        for i in range(len(Y)):
            if np.random.random() < prob:
                new_y = np.random.randint(num_classes)
                while new_y == train_y[i]:
                    new_y = np.random.randint(num_classes)
                Y[i] = new_y

        path = f'./data/{data}-train-y-sym-{prob}.npy'
        np.save(path, Y)
else:
    for prob in probs:
        Y = train_y.copy()
        bound = max(Y)
        noises = np.random.binomial(1, prob, (len(Y),)) * np.random.uniform(-5*bound, 5*bound, (len(Y),))
        Y += noises
        
        path = f'./data/{data}-train-y-{prob}'
        np.save(path, Y)
        