from Network import network, histoscore
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
import pickle
from os.path import isfile
import time
import numpy as np
import tonic

def tic():
    global ttic
    ttic = time.time()
def toc():
    print(f'Done in {time.time() - ttic:.3f} s')

class AERtoVectDataset(Dataset):
    """makes a dataset allowing aer_to_vect() transform from tonic
    """
    classes = [
        "0 - zero",
        "1 - one",
        "2 - two",
        "3 - three",
        "4 - four",
        "5 - five",
        "6 - six",
        "7 - seven",
        "8 - eight",
        "9 - nine",
    ]
    sensor_size = [34, 34]
    ordering = "xytp"
    
    def __init__(self, tensors, digind, transform=None, nb_pola=2):
        self.X_train, self.y_train = tensors
        assert (self.X_train.shape[0] == len(self.y_train))
        self.transform = transform
        self.digind = digind

    def __getitem__(self, index):
        events = self.X_train[self.digind[index]:self.digind[index+1]]
        if self.transform:
            events = self.transform(events, self.sensor_size, self.ordering)
        target = self.y_train[self.digind[index]]
        return events, target

    def __len__(self):
        return len(self.digind)-1
    
class LRtorch(torch.nn.Module):
    #torch.nn.Module -> Base class for all neural network modules
    def __init__(self, N, n_classes, bias=True):
        super(LRtorch, self).__init__() 
        self.linear = torch.nn.Linear(N, n_classes, bias=bias)
        self.nl = torch.nn.Softmax(dim=1)

    def forward(self, factors):
        return self.nl(self.linear(factors))

def get_loader(name, nb_digit, train, filt, tau, nblay, nbclust, sigma, homeinv, jitter, timestr):

    hotshom, homeotest = netparam(name, filt, tau, nblay, nbclust, sigma, homeinv, jitter, timestr)
    stream = hotshom.running(homeotest = homeotest, nb_digit=nb_digit, train=train, LR=True)

    # get indices for transitions from one digit to another 
    def getdigind(stream):
        t = np.array(stream[2])
        newdig = [0]
        for i in range(len(t)-1):
            if t[i]>t[i+1]:
                newdig.append(i+1)
        newdig.append(i)
        return newdig

    events_train = np.zeros([len(stream[2]), 4])
    ordering = 'xytp'
    for i in range(4):
        events_train[:, i] = stream[i][:]

    X_train = events_train.astype(int)
    y_train = stream[4]
    digind_train = getdigind(stream)

    nb_pola = stream[-1]
    # Dataset w/o any tranformations
    train_dataset = AERtoVectDataset(tensors=(X_train, y_train), digind=digind_train,
                                        transform=tonic.transforms.AERtoVector(nb_pola = nb_pola))
    #train_loader = torch.utils.data.DataLoader(train_dataset_normal, batch_size=1)
    name_net = hotshom.get_fname()
    
    return train_dataset, nb_pola, name_net


def fit_data(name,
            dataset, 
            nb_digit,
            nb_pola,
            learning_rate,
            num_epochs,
            betas,
            verbose=False, #**kwargs
        ):
    
    if isfile(name):
        with open(name, 'rb') as file:
            logistic_model, losses = pickle.load(file)
    else:
    
        torch.set_default_tensor_type("torch.DoubleTensor")
        criterion = torch.nn.BCELoss(reduction="mean")
        amsgrad = True #or False gives similar results

        generator = torch.Generator().manual_seed(42)
        sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=nb_digit, generator=generator)
        loader = tonic.datasets.DataLoader(dataset, sampler=sampler)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'device -> {device}')

        N = dataset.sensor_size[0]*dataset.sensor_size[1]*nb_pola
        n_classes = len(dataset.classes)
        logistic_model = LRtorch(N, n_classes)
        logistic_model = logistic_model.to(device)
        logistic_model.train()
        optimizer = torch.optim.Adam(
            logistic_model.parameters(), lr=learning_rate, betas=betas, amsgrad=amsgrad
        )
        print('quoi?')
        for epoch in range(int(num_epochs)):
            losses = []
            for X, label in loader:
                X, label = X.to(device), label.to(device)
                X, label = X.squeeze(0), label.squeeze(0) # just one digit = one batch

                outputs = logistic_model(X)

                n_events = X.shape[0]
                labels = label*torch.ones(n_events).type(torch.LongTensor).to(device)
                labels = torch.nn.functional.one_hot(labels, num_classes=n_classes).type(torch.DoubleTensor).to(device)

                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            if verbose and (epoch % (num_epochs // 32) == 0):
                print(f"Iteration: {epoch} - Loss: {np.mean(losses):.5f}")
                
        with open(name, 'wb') as file:
            pickle.dump([logistic_model, losses], file, pickle.HIGHEST_PROTOCOL)
            
    return logistic_model, losses

def predict_data(test_set, model, 
            verbose=False, **kwargs
        ):
    
    with torch.no_grad():

        generator=torch.Generator().manual_seed(42)
        sampler = torch.utils.data.RandomSampler(test_set, replacement=True, num_samples=nb_test, generator=generator)
        loader = tonic.datasets.DataLoader(test_set, sampler=sampler)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logistic_model = model.to(device)

        pred_target, true_target = [], []

        for X, label in loader:
            X = X.to(device)
            X, label = X.squeeze(0), label.squeeze(0)

            n_events = X.shape[0]
            labels = label*torch.ones(n_events).type(torch.LongTensor)

            outputs = logistic_model(X)

            pred_target.append(torch.argmax(outputs, dim=1).cpu().numpy())
            true_target.append(labels.numpy())

    return pred_target, true_target


def netparam(name, filt, tau, nblay, nbclust, sigma, homeinv, jitter, timestr):
    if name=='hots':
        homeo = False
        homeotest = False
        krnlinit = 'first'
        hotshom = network(krnlinit=krnlinit, filt=filt, tau=tau, nblay=nblay, nbclust=nbclust, homeo=homeo, sigma=sigma, homeinv=homeinv, jitter=jitter, timestr=timestr)
        hotshom = hotshom.learning1by1()
    elif name=='homhots':
        homeo = True
        homeotest = False
        krnlinit = 'rdn'
        hotshom = network(krnlinit=krnlinit, filt=filt, tau=tau, nblay=nblay, nbclust=nbclust, homeo=homeo, sigma=sigma, homeinv=homeinv, jitter=jitter, timestr=timestr)
        hotshom = hotshom.learningall()
    elif name=='fullhom':
        homeo = True
        homeotest = True
        krnlinit = 'rdn'
        hotshom = network(krnlinit=krnlinit, filt=filt, tau=tau, nblay=nblay, nbclust=nbclust, homeo=homeo, sigma=sigma, homeinv=homeinv, jitter=jitter, timestr=timestr)
        hotshom = hotshom.learningall()
    elif name=='onlyonline':
        homeo = False
        homeotest = False
        krnlinit = 'rdn'
        hotshom = network(krnlinit=krnlinit, filt=filt, tau=tau, nblay=nblay, nbclust=nbclust, homeo=homeo, sigma=sigma, homeinv=homeinv, jitter=jitter, timestr=timestr)
        hotshom = hotshom.learningall()
    return hotshom, homeotest

def runjit(timestr, name, filt, tau, nblay, nbclust, sigma, homeinv, jitter, jit_s, jit_t, nb_train, nb_test, verbose=False):
    
    hotshom, homeotest = netparam(name, filt, tau, nblay, nbclust, sigma, homeinv, jitter, timestr)
    
    f_name = f'../Records/EXP_03_NMNIST/{hotshom.get_fname()}_jitter_histo_{nb_train}_{nb_test}.pkl'
    if isfile(f_name):
        with open(f_name, 'rb') as file:
            score_T, jit_t, score_S, jit_s = pickle.load(file)
    else:
        trainhistomap = hotshom.running(homeotest=homeotest, nb_digit = nb_train)
        score_S = []
        score_T = []
        for i in jit_s:
            i = round(i,1)
            jitonic = [None,i]
            testhistomap = hotshom.running(homeotest = homeotest, train=False, nb_digit=nb_test, jitonic=jitonic)
            JS_score = histoscore(trainhistomap,testhistomap, verbose = verbose)
            print(f'loading... - spatial jitter = {i} - score = {JS_score}',end='\r')
            score_S.append(JS_score)

        for j in jit_t:
            j = round(j,1)
            jitonic = [j,None]
            testhistomap = hotshom.running(homeotest = homeotest, train=False, nb_digit=nb_test, jitonic=jitonic)
            JS_score = histoscore(trainhistomap,testhistomap, verbose = verbose)
            print(f'loading... - temporal jitter = {j} - score = {JS_score}',end='\r')
            score_T.append(JS_score)

        with open(f_name, 'wb') as file:
            pickle.dump([score_T, jit_t, score_S, jit_s], file, pickle.HIGHEST_PROTOCOL)
        
    return score_T, jit_t, score_S, jit_s 
