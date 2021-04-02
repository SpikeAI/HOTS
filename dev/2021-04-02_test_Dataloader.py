import time
pin_memory = True
print('pin_memory is', pin_memory)

import tonic
dataset = tonic.datasets.NMNIST(save_to='../Data/', download=False)

print('> Loading vanilla NMNIST')
for num_workers in range(20): 
    loader = tonic.datasets.DataLoader(dataset, num_workers=num_workers, pin_memory=pin_memory)
    start = time.time()
    for epoch in range(1, 5):
        for i, data in enumerate(loader):
            pass
    end = time.time()
    print("Finish with:{} second, num_workers={}".format(end - start, num_workers))