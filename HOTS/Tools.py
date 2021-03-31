from Network import network, histoscore
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
import pickle
from os.path import isfile
import time
import numpy as np
import tonic
import matplotlib.pyplot as plt
from tqdm import tqdm

def tic():
    global ttic
    ttic = time.time()
def toc():
    print(f'Done in {time.time() - ttic:.3f} s')
    
#_________________________________FOR_LR_ON_HOTS____________________________________________
#___________________________________________________________________________________________


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

def get_loader(name, path, nb_digit, train, filt, tau, nbclust, sigma, homeinv, jitter, timestr, dataset, R, jitonic=[None,None]):

    if name=='raw':
        name_net = f'{path}{timestr}_{name}_LR_{nb_digit}.pkl'
        download = False
        train_dataset = tonic.datasets.NMNIST(save_to='../Data/',
                                  train=train, download=download,
                                  transform=tonic.transforms.AERtoVector()
                                 )
        nb_pola = 2
    else:
        hotshom, homeotest = netparam(name, filt, tau, nbclust, sigma, homeinv, jitter, timestr, dataset, R)
        stream = hotshom.running(homeotest = homeotest, nb_digit=nb_digit, train=train, outstyle='LR')
        # get indices for transitions from one digit to another
        
        #TODO: save in event stream from network.running directly
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
        name_net = f'{path}{hotshom.get_fname()}_LR_{nb_digit}.pkl'
    
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
        print('loading existing model')
        print(name)
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
        pbar = tqdm(total=int(num_epochs))
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
                
            pbar.update(1)
            #if verbose and (epoch % (num_epochs // 32) == 0):
            #    print(f"Iteration: {epoch} - Loss: {np.mean(losses):.5f}")
        pbar.close()
        with open(name, 'wb') as file:
            pickle.dump([logistic_model, losses], file, pickle.HIGHEST_PROTOCOL)
            
    return logistic_model, losses

def predict_data(test_set, model, nb_test,
            verbose=False, **kwargs
        ):
    
    with torch.no_grad():

        generator=torch.Generator().manual_seed(42)
        sampler = torch.utils.data.RandomSampler(test_set, replacement=True, num_samples=nb_test, generator=generator)
        loader = tonic.datasets.DataLoader(test_set, sampler=sampler)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logistic_model = model.to(device)
        
        pbar = tqdm(total=len(test_set.digind)-1)
        pred_target, true_target = [], []
        for X, label in loader:
            X = X.to(device)
            X, label = X.squeeze(0), label.squeeze(0)

            n_events = X.shape[0]
            labels = label*torch.ones(n_events).type(torch.LongTensor)

            outputs = logistic_model(X)

            pred_target.append(torch.argmax(outputs, dim=1).cpu().numpy())
            true_target.append(labels.numpy())
            pbar.update(1)
        pbar.close()

    return pred_target, true_target

#___________________________________________________________________________________________
#___________________________________________________________________________________________


#_______________________________TO_RUN_HOTS_________________________________________________
#___________________________________________________________________________________________

def netparam(name, filt, tau, nbclust, sigma, homeinv, jitter, timestr, dataset, R, nb_learn=10):
    print(dataset)
    if name=='hots':
        homeo = False
        homeotest = False
        krnlinit = 'first'
        hotshom = network(krnlinit=krnlinit, filt=filt, tau=tau, R=R, nbclust=nbclust, homeo=homeo, sigma=sigma, homeinv=homeinv, jitter=jitter, timestr=timestr)
        hotshom = hotshom.learning1by1(dataset=dataset, nb_digit = nb_learn)
    elif name=='homhots':
        homeo = True
        homeotest = False
        krnlinit = 'rdn'
        hotshom = network(krnlinit=krnlinit, filt=filt, tau=tau, R=R, nbclust=nbclust, homeo=homeo, sigma=sigma, homeinv=homeinv, jitter=jitter, timestr=timestr)
        hotshom = hotshom.learningall(dataset=dataset, nb_digit = nb_learn)
    elif name=='fullhom':
        homeo = True
        homeotest = True
        krnlinit = 'rdn'
        hotshom = network(krnlinit=krnlinit, filt=filt, tau=tau, R=R, nbclust=nbclust, homeo=homeo, sigma=sigma, homeinv=homeinv, jitter=jitter, timestr=timestr)
        hotshom = hotshom.learningall(dataset=dataset, nb_digit = nb_learn)
    elif name=='onlyonline':
        homeo = False
        homeotest = False
        krnlinit = 'rdn'
        hotshom = network(krnlinit=krnlinit, filt=filt, tau=tau, R=R, nbclust=nbclust, homeo=homeo, sigma=sigma, homeinv=homeinv, jitter=jitter, timestr=timestr)
        hotshom = hotshom.learningall(dataset=dataset, nb_digit = nb_learn)
    return hotshom, homeotest

def runjit(timestr, name, path, filt, tau, nbclust, sigma, homeinv, jitter, jit_s, jit_t, nb_train, nb_test, dataset, verbose=False):
    
    hotshom, homeotest = netparam(name, filt, tau, nbclust, sigma, homeinv, jitter, timestr, dataset)
    
    f_name = f'{path}{hotshom.get_fname()}_jitter_histo_{nb_train}_{nb_test}.pkl'
    if isfile(f_name):
        with open(f_name, 'rb') as file:
            score_T, jit_t, score_S, jit_s = pickle.load(file)
    else:
        trainhistomap = hotshom.running(homeotest=homeotest, nb_digit = nb_train, dataset=dataset)
        score_S = []
        score_T = []
        for i in jit_s:
            i = round(i,2)
            jitonic = [None,i]
            testhistomap = hotshom.running(homeotest = homeotest, train=False, nb_digit=nb_test, jitonic=jitonic, dataset=dataset)
            JS_score = histoscore(trainhistomap,testhistomap, verbose = verbose)
            print(f'loading... - spatial jitter = {i} - score = {JS_score}',end='\r')
            score_S.append(JS_score)

        for j in jit_t:
            j = round(j,0)
            jitonic = [j,None]
            testhistomap = hotshom.running(homeotest = homeotest, train=False, nb_digit=nb_test, jitonic=jitonic, dataset=dataset)
            JS_score = histoscore(trainhistomap,testhistomap, verbose = verbose)
            print(f'loading... - temporal jitter = {j} - score = {JS_score}',end='\r')
            score_T.append(JS_score)

        with open(f_name, 'wb') as file:
            pickle.dump([score_T, jit_t, score_S, jit_s], file, pickle.HIGHEST_PROTOCOL)
            
    if verbose:
        jit_t, jit_s = np.sqrt(jit_t), np.sqrt(jit_s)
        plt.subplot(1,2,1)
        plt.plot(jit_s, score_S,'.')
        plt.title('accuracy as a function of \n std of spatial jitter')
        plt.subplot(1,2,2)
        plt.plot(jit_t, score_T,'.')
        plt.title('accuracy as a function of \n std of temporal jitter')
        plt.show()
        jit_t, jit_s = jit_t**2, jit_s**2
    return score_T, jit_t, score_S, jit_s

def classification_results(pred_target, true_target, nb_test, verbose=False):
    accuracy = []
    onlinac = np.zeros(10000) # vector that has size bigger than all event stream size
    for pred_target_, true_target_ in zip(pred_target, true_target):
        accuracy.append(np.mean(pred_target_ == true_target_))
        fill_pred = pred_target_[-1]*np.ones(len(onlinac)-len(pred_target_))
        fill_true = true_target_[-1]*np.ones(len(onlinac)-len(pred_target_))
        #fill_pred[:] = np.NaN
        #fill_true[:] = np.NaN
        pred_target_ = np.concatenate([pred_target_,fill_pred])
        true_target_ = np.concatenate([true_target_,fill_true])
        onlinac+=(pred_target_ == true_target_)
        
    if verbose:
        print(f'{np.mean(accuracy)=:.3f}')
        plt.plot(onlinac[:5000]/nb_test);
        plt.xlabel('number of events');
        plt.ylabel('online accuracy');
        plt.title('LR classification results evolution as a function of the number of events');
    
    return np.mean(accuracy), onlinac

#___________________________________________________________________________________________
#___________________________________________________________________________________________


#___________________________FIT_WITH_SIGMOID________________________________________________
#___________________________________________________________________________________________

def fit_jitter(param,
    theta,
    y,
    learning_rate=0.005,
    batch_size=256,  # gamma=gamma,
    num_epochs=2 ** 9 + 1,
    betas=(0.9, 0.999),
    verbose=False, **kwargs
):
    
    torch.set_default_tensor_type("torch.DoubleTensor")
    criterion = torch.nn.BCELoss(reduction="mean")

    class LRModel_jitter(torch.nn.Module):
        def __init__(self, bias=True, logit0=0, jitter0=0, log_wt=torch.log(2*torch.ones(1)), n_classes=10):
            super(LRModel_jitter, self).__init__()
            self.jitter0 = torch.nn.Parameter(jitter0 * torch.ones(1))
            self.logit0 = torch.nn.Parameter(logit0 * torch.ones(1))
            self.log_wt = torch.nn.Parameter(log_wt * torch.ones(1))
            self.n_classes = n_classes

        def forward(self, jitter):
            p0 = torch.sigmoid(self.logit0)
            p = torch.sigmoid((self.jitter0-jitter)/torch.exp(self.log_wt))
            out = 1/self.n_classes + (1 - p0 - 1/self.n_classes) * p
            #out = 1-p0 / 2 + (1 - p0) * torch.sigmoid((jitter-self.jitter0 )/torch.exp(self.log_wt))
            return out
        
    amsgrad = True  # gives similar results

    Theta, labels = torch.Tensor(theta[:, None]), torch.Tensor(y[:, None])
    loader = DataLoader(
        TensorDataset(Theta, labels), batch_size=batch_size, shuffle=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logistic_model = LRModel_jitter(logit0=param[0],jitter0=param[1],log_wt=torch.from_numpy(param[2]))
    logistic_model = logistic_model.to(device)
    logistic_model.train()
    optimizer = torch.optim.Adam(
        logistic_model.parameters(), lr=learning_rate, betas=betas, amsgrad=amsgrad
    )
    for epoch in range(int(num_epochs)):
        logistic_model.train()
        losses = []
        for Theta_, labels_ in loader:
            Theta_, labels_ = Theta_.to(device), labels_.to(device)
            outputs = logistic_model(Theta_)
            loss = criterion(outputs, labels_)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        if verbose and (epoch % (num_epochs // 32) == 0):
            print(f"Iteration: {epoch} - Loss: {np.sum(losses)/len(theta):.5f}")

    logistic_model.eval()
    Theta, labels = torch.Tensor(theta[:, None]).to(device), torch.Tensor(y[:, None]).to(device)
    outputs = logistic_model(Theta)
    loss = criterion(outputs, labels).item() / len(theta)
    fit = torch.squeeze(logistic_model(Theta)).cpu()
    return logistic_model, loss, fit

def signumber(x,nb):
    c = np.log10(x)
    C = int(np.floor(c))
    b = x/10**C
    B = np.round(b,nb)
    if not C:
        num = f'{B}'
    elif C==-1:
        num = f'{np.round(B/10,nb+1)}'
    elif C==1:
        num = f'{np.round(B*10,nb-1)}'
    else:
        num = f'{B}e{C}'
    return num

#___________________________________________________________________________________________
#___________________________________________________________________________________________


#___________________________HISTOGRAM_CLASSIFICATION________________________________________
#___________________________________________________________________________________________

def histoscore_lagorce(trainmap,testmap, verbose = True):
    
    bhat_score = accuracy_lagorce(trainmap, testmap, 'bhatta')
    norm_score = accuracy_lagorce(trainmap, testmap, 'norm')
    eucl_score = accuracy_lagorce(trainmap, testmap, 'eucli')
    KL_score = accuracy_lagorce(trainmap,testmap,'KL')
    JS_score = accuracy_lagorce(trainmap,testmap,'JS')
    if verbose:
        print(47*'-'+'SCORES'+47*'-')
        print(f'Classification scores with HOTS measures: bhatta = {np.round(bhat_score*100)}% - eucli = {np.round(eucl_score*100)}% - norm = {np.round(norm_score*100)}%')
        print(f'Classification scores with entropy: Kullback-Leibler = {np.round(KL_score*100)}% - Jensen-Shannon = {np.round(JS_score*100)}%')
        print(100*'-')
    return bhat_score, norm_score, eucl_score, KL_score, JS_score

def histoscore(trainmap,testmap, verbose = True):
    bhat_score = accuracy(trainmap, testmap, 'bhatta')
    norm_score = accuracy(trainmap, testmap, 'norm')
    eucl_score = accuracy(trainmap, testmap, 'eucli')
    KL_score = accuracy(trainmap,testmap,'KL')
    JS_score = accuracy(trainmap,testmap,'JS')
    kNN_6 = knn(trainmap,testmap,6)
    kNN_3 = knn(trainmap,testmap,3)
    if verbose:
        print(47*'-'+'SCORES'+47*'-')
        print(f'Classification scores with HOTS measures: bhatta = {np.round(bhat_score*100)}% - eucli = {np.round(eucl_score*100)}% - norm = {np.round(norm_score*100)}%')
        print(f'Classification scores with entropy: Kullback-Leibler = {np.round(KL_score*100)}% - Jensen-Shannon = {np.round(JS_score*100)}%')
        print(f'Classification scores with k-NN: 3-NN = {np.round(kNN_3*100)}% - 6-NN = {np.round(kNN_6*100)}%')
        print(100*'-')
    return bhat_score, norm_score, eucl_score, KL_score, JS_score, kNN_3, kNN_6

def knn(trainmap,testmap,k):
    from sklearn.neighbors import KNeighborsClassifier

    X_train = np.array([trainmap[i][1]/np.sum(trainmap[i][1]) for i in range(len(trainmap))]).reshape(len(trainmap),len(trainmap[0][1]))
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,[trainmap[i][0] for i in range(len(trainmap))])
    accuracy = 0
    for i in range(len(testmap)):
        if knn.predict([testmap[i][1]/np.sum(testmap[i][1])])==testmap[i][0]:
            accuracy += 1
    return accuracy/len(testmap)

def EuclidianNorm(hist1,hist2):
    return np.linalg.norm(hist1-hist2)

def NormalizedNorm(hist1,hist2):
    return np.linalg.norm(hist1-hist2)/(np.linalg.norm(hist1)*np.linalg.norm(hist2))

def BattachaNorm(hist1, hist2):
    return -np.log(np.sum(np.sqrt(hist1*hist2)))

def KullbackLeibler(hist_test, hist_train):
    return np.sum(hist_test*np.log(hist_test/hist_train))

def JensenShannon(hist1, hist2):
    hist3 = (hist1+hist2)*0.5
    return (KullbackLeibler(hist1,hist3)+KullbackLeibler(hist2,hist3))*0.5

def accuracy_lagorce(trainmap,testmap,measure):
    accuracy=0
    total = 0
    for i in range(len(testmap)):
        dist = np.zeros([trainmap.shape[0]])
        histest = testmap[i][1]/np.sum(testmap[i][1])
        for k in range(trainmap.shape[0]):
            histrain = trainmap[k,:]/np.sum(trainmap[k,:])
            if measure=='bhatta':
                dist[k] = BattachaNorm(histest,histrain)
            elif measure=='eucli':
                dist[k] = EuclidianNorm(histest,histrain)
            elif measure=='norm':
                dist[k] = NormalizedNorm(histest,histrain)
            elif measure == 'KL':
                dist[k] = KullbackLeibler(histest,histrain)
            elif measure == 'JS':
                dist[k] = JensenShannon(histest,histrain)
        if testmap[i][0]== np.argmin(dist):
            accuracy+=1
        total+=1
    return accuracy/total

def accuracy(trainmap,testmap,measure):
    accuracy=0
    total = 0
    for i in range(len(testmap)):
        dist = np.zeros(len(trainmap))
        histest = testmap[i][1]/np.sum(testmap[i][1])
        for k in range(len(trainmap)):
            histrain = trainmap[k][1]/np.sum(trainmap[k][1])
            if measure=='bhatta':
                dist[k] = BattachaNorm(histest,histrain)
            elif measure=='eucli':
                dist[k] = EuclidianNorm(histest,histrain)
            elif measure=='norm':
                dist[k] = NormalizedNorm(histest,histrain)
            elif measure == 'KL':
                dist[k] = KullbackLeibler(histest,histrain)
            elif measure == 'JS':
                dist[k] = JensenShannon(histest,histrain)
        if testmap[i][0]==trainmap[np.argmin(dist)][0]:
            accuracy+=1
        total+=1
    return accuracy/total