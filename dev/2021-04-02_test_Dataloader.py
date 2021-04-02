import time
import numpy as np
pin_memory = True
print('pin_memory is', pin_memory)

import tonic
dataset = tonic.datasets.NMNIST(save_to='../Data/',
                                  train=False, download=False,
                                 )

print('> Loading vanilla NMNIST')
for num_workers in range(5): 
    loader = tonic.datasets.DataLoader(dataset, num_workers=num_workers, pin_memory=pin_memory)
    start = time.time()
    for epoch in range(1, 5):
        for i, data in enumerate(loader):
            pass
    end = time.time()
    print("Finish with:{} second, num_workers={}".format(np.round(end - start,2), num_workers))
    
dataset = tonic.datasets.NMNIST(save_to='../Data/',
                                  train=False, download=False,
                                  transform=tonic.transforms.AERtoVector()
                                 )

print('> Loading and transform NMNIST')
for num_workers in range(5): 
    loader = tonic.datasets.DataLoader(dataset, num_workers=num_workers, pin_memory=pin_memory)
    start = time.time()
    for epoch in range(1, 5):
        for i, data in enumerate(loader):
            pass
    end = time.time()
    print("Finish with:{} second, num_workers={}".format(np.round(end - start,2), num_workers))