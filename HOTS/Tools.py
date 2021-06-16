from Network import network, LoadFromMat
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, SubsetRandomSampler
import pickle
from os.path import isfile
import time
import numpy as np
import tonic
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import beta

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
    ordering = "xytp"
    
    def __init__(self, tensors, digind, name, transform=None, nb_pola=2):
        self.X_train, self.y_train = tensors
        assert (self.X_train.shape[0] == len(self.y_train))
        self.transform = transform
        self.digind = digind
        if name=='nmnist':
            self.classes = [
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
            self.sensor_size = [34, 34]
        elif name=='poker':
            self.classes = ["cl", "he", "di", "sp"]
            self.sensor_size = [35, 35]
        elif name=='cars':
            self.classes = ["background", "car"]
            self.sensor_size = [120, 100]
        elif name=='barrel':
            self.sensor_size = [32, 32]
            self.classes = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","0","1","2","3","4","5","6","7","8","9"]

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
    
def getdigind(t, l):
    newdig = [0]
    i = 0
    for i in range(len(t)-1):
        if t[i]>t[i+1] or l[i]!=l[i+1]:
            newdig.append(i+1)
    if i>0:
        newdig.append(i+1)
    return newdig

def get_loader(name, 
               path, 
               nb_digit, 
               train, 
               filt, tau, 
               nbclust, 
               sigma, 
               homeinv, 
               jitter, 
               timestr, 
               dataset, 
               R, 
               num_workers,
               tau_cla,
               subset_size = None, 
               jitonic=[None,None], 
               ds_ev = None, 
               sample_space = 1,
               verbose = True):

    if name=='raw':
        download = False
        if jitonic[1] is not None:
            print(f'spatial jitter -> std = {np.sqrt(jitonic[1])}')
            transform = tonic.transforms.Compose([tonic.transforms.SpatialJitter(variance_x=jitonic[1], variance_y=jitonic[1], sigma_x_y=0, integer_coordinates=True, clip_outliers=True), tonic.transforms.AERtoVector(sample_event=ds_ev, tau = tau_cla, sample_space=sample_space)])

        if jitonic[0] is not None:
            print(f'time jitter -> std = {jitonic[0]}')
            transform = tonic.transforms.Compose([tonic.transforms.TimeJitter(std=jitonic[0], integer_jitter=False, clip_negative=True, sort_timestamps=True), tonic.transforms.AERtoVector(sample_event=ds_ev, tau = tau_cla, sample_space=sample_space)])

        if jitonic == [None,None]:
            print('no jitter')
            transform = tonic.transforms.Compose([tonic.transforms.AERtoVector(sample_event=ds_ev, tau = tau_cla, sample_space=sample_space)])
        
        
        if dataset == 'nmnist':
            train_dataset = tonic.datasets.NMNIST(save_to='../Data/',
                                  train=train, download=download,
                                  transform=transform
                                 )
            time_dataset = tonic.datasets.NMNIST(save_to='../Data/',
                                  train=train, download=download,
                                 )
        elif dataset == 'poker':
            train_dataset = tonic.datasets.POKERDVS(save_to='../Data/',
                                  train=train, download=download,
                                  transform=transform
                                 )
            time_dataset = tonic.datasets.POKERDVS(save_to='../Data/',
                                  train=train, download=download,
                                 )
        elif dataset == 'cars':
            train_dataset = tonic.datasets.NCARS(save_to='../Data/',
                                  train=train, download=download,
                                  transform=transform
                                 )
            time_dataset = tonic.datasets.NCARS(save_to='../Data/',
                                  train=train, download=download,
                                 )
        nb_pola = 2
        time_scale = []
        if subset_size is not None:
            subset_indices = []
            for i in range(len(train_dataset.classes)):
                all_ind = np.where(np.array(train_dataset.targets)==i)[0]
                subset_indices += all_ind[:subset_size//len(train_dataset.classes)].tolist()
            g_cpu = torch.Generator()
            subsampler = SubsetRandomSampler(subset_indices, g_cpu)
            loader = tonic.datasets.DataLoader(train_dataset, batch_size=1, shuffle=False, sampler=subsampler, num_workers=num_workers)
        else:
            if train:
                generator = torch.Generator().manual_seed(42)
                sampler = torch.utils.data.RandomSampler(train_dataset, replacement=True, num_samples=len(train_dataset), generator=generator)
                loader = tonic.datasets.DataLoader(train_dataset, sampler=sampler, num_workers=num_workers, shuffle=False)
            else:
                loader = tonic.datasets.DataLoader(train_dataset, num_workers=num_workers, shuffle=False)
                time_loader = tonic.datasets.DataLoader(time_dataset, num_workers=num_workers, shuffle=False)
                for events, label in time_loader:
                    time_scale.append(events[0,:,time_dataset.ordering.find("t")].numpy())
    else:
        hotshom, homeotest = netparam(name, filt, tau, nbclust, sigma, homeinv, jitter, timestr[:10], dataset, R, verbose=verbose)
        if not train:
            hotshom.date = timestr

        stream = hotshom.running(homeotest = homeotest, nb_digit=nb_digit, train=train, dataset = dataset, jitonic=jitonic, outstyle='LR', subset_size=subset_size, verbose = verbose)
        
        #TODO: save in event stream from network.running directly

        events_train = np.zeros([len(stream[2]), 4])
            
        for i in range(4):
            events_train[:,i] = stream[i][:]

        X_train = events_train.astype(int)
        y_train = stream[4]

        digind_train = getdigind(np.array(X_train[:,2]), y_train)

        nb_pola = stream[-1]
        train_dataset = AERtoVectDataset(tensors=(X_train, y_train), digind=digind_train, name = dataset,transform=tonic.transforms.Compose([tonic.transforms.AERtoVector(nb_pola = nb_pola, sample_event= ds_ev, tau = tau_cla, sample_space=sample_space)]))
        
        time_scale = []
        if train:
            generator = torch.Generator().manual_seed(42)
            sampler = torch.utils.data.RandomSampler(train_dataset, replacement=True, num_samples=nb_digit, generator=generator)
            loader = tonic.datasets.DataLoader(train_dataset, sampler=sampler, num_workers=num_workers, shuffle=False)
        else:
            loader = tonic.datasets.DataLoader(train_dataset, num_workers=num_workers, shuffle=False)
            for i in range(len(digind_train)-1):
                time_scale.append(np.array(X_train[digind_train[i]:digind_train[i+1],2]))
        
    return loader, train_dataset, nb_pola, time_scale

def get_loader_barrel(name, 
               path, 
               train, 
               filt, 
               tau, 
               nbclust, 
               sigma, 
               homeinv, 
               jitter, 
               timestr, 
               dataset, 
               R, 
               num_workers,
               tau_cla,
               ds_ev = 1,
               sample_space = 1,
               verbose = True):
    
    class_data = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6,
    "H": 7,
    "I": 8,
    "J": 9,
    "K": 10,
    "L": 11,
    "M": 12,
    "N": 13,
    "O": 14,
    "P": 15,
    "Q": 16,
    "R": 17,
    "S": 18,
    "T": 19,
    "U": 20,
    "V": 21,
    "W": 22,
    "X": 23,
    "Y": 24,
    "Z": 25,
    "0": 26,
    "1": 27,
    "2": 28,
    "3": 29,
    "4": 30,
    "5": 31,
    "6": 32,
    "7": 33,
    "8": 34,
    "9": 35,
}
    
    def getdigind_barrel(t):
            newdig = [0]
            for i in range(len(t)-1):
                if t[i]>t[i+1]:
                    newdig.append(i+1)
            newdig.append(i)
            return newdig
        
    if train:
        nb_digit = 36
    else:
        nb_digit = 40

    if name=='raw':
        time_scale = []
        if train:
            path = "../Data/alphabet_ExtractedStabilized.mat"
            image_list=list(np.arange(0, 36))
            address, time, polarity, list_pola = LoadFromMat(path, image_number=image_list)
            with open('../Data/alphabet_label.pkl', 'rb') as file:
                label_list = pickle.load(file)
            label = label_list[:36]
        else:
            path = "../Data/alphabet_ExtractedStabilized.mat"
            image_list=list(np.arange(36, 76))
            address, time, polarity, list_pola = LoadFromMat(path, image_number=image_list)
            with open('../Data/alphabet_label.pkl', 'rb') as file:
                label_list = pickle.load(file)
            label = label_list[36:76]
        nb_pola = 2
        
        X_train = np.zeros([len(time), 4])
        X_train[:,2] = time*1e6
        X_train[:,0] = address[:,0]
        X_train[:,1] = address[:,1]
        X_train[:,3] = polarity
        X_train = X_train.astype(int)
        
        y_train = []
        for i in range(len(label)):
            for j in range(label[i][1]+1):
                y_train.append(class_data[label[i][0]])
        y_train = np.array(y_train)
        
        digind_train = getdigind_barrel(np.array(X_train[:,2]))
         
        for i in range(len(digind_train)-1):
            time_scale.append(np.array(X_train[digind_train[i]:digind_train[i+1],2]))
        
    else:
        if name == 'homhots':
            if train:
                f_name = '../Records/EXP_01_LagorceKmeans/train/2020-12-01_lagorce_rdn_None_True_[0.25, 1]_[4, 8, 16]_[10.0, 100.0, 1000.0]_[2, 4, 8]_False_36_None_LR.pkl'
            else: 
                f_name = '../Records/EXP_01_LagorceKmeans/test/2020-12-01_lagorce_rdn_None_True_[0.25, 1]_[4, 8, 16]_[10.0, 100.0, 1000.0]_[2, 4, 8]_False_40_None_LR.pkl'
            with open(f_name,'rb') as file:
                stream = pickle.load(file)
        elif name == 'hots':
            if train:
                f_name = '../Records/EXP_01_LagorceKmeans/train/2020-12-01_lagorce_first_None_False_[0.25, 1]_[4, 8, 16]_[10.0, 100.0, 1000.0]_[2, 4, 8]_False_36_None_LR.pkl'
            else:
                f_name = '../Records/EXP_01_LagorceKmeans/test/2020-12-01_lagorce_first_None_False_[0.25, 1]_[4, 8, 16]_[10.0, 100.0, 1000.0]_[2, 4, 8]_False_40_None_LR.pkl'
            with open(f_name,'rb') as file:
                stream = pickle.load(file)

        events_train = np.zeros([len(stream[2]), 4])
            
        for i in range(4):
            events_train[:,i] = stream[i][:]
            
        events_train[:,2]*=1e6
        X_train = events_train.astype(int)
        y_train = stream[4]

        digind_train = getdigind_barrel(np.array(X_train[:,2]))
        nb_pola = stream[-1]
        
    train_dataset = AERtoVectDataset(tensors=(X_train, y_train), digind=digind_train, name = dataset,transform=tonic.transforms.AERtoVector(nb_pola = nb_pola, sample_event= ds_ev, tau = tau_cla, sample_space=sample_space))
    
    generator = torch.Generator().manual_seed(42)
    sampler = torch.utils.data.RandomSampler(train_dataset, replacement=True, num_samples=nb_digit, generator=generator)
    loader = tonic.datasets.DataLoader(train_dataset, sampler=sampler, num_workers=num_workers, shuffle=False)
    
    time_scale = []
    if train:
        generator = torch.Generator().manual_seed(42)
        sampler = torch.utils.data.RandomSampler(train_dataset, replacement=True, num_samples=nb_digit, generator=generator)
        loader = tonic.datasets.DataLoader(train_dataset, sampler=sampler, num_workers=num_workers, shuffle=False)
    else:
        loader = tonic.datasets.DataLoader(train_dataset, num_workers=num_workers, shuffle=False)
        for i in range(len(digind_train)-1):
            time_scale.append(np.array(X_train[digind_train[i]:digind_train[i+1],2]))
        
    return loader, train_dataset, nb_pola, time_scale

def fit_data(name,
             timestr,
             path,
             filt,
             tau,
             R,
             nbclust,
             sigma,
             homeinv,
             jitter,
             dataset,
             nb_digit,
             ds_ev,
             learning_rate,
             num_epochs,
             betas,
             tau_cla,
             jitonic = [None, None],
             subset_size = None,
             num_workers = 0,
             nb_learn = 10,
             sample_space=1,
             verbose=False, #**kwargs
        ):
    
    path = path+'models/'
    
    if name=='raw':
        name_model = f'{path}{timestr}_{name}_LR_{nb_digit}_{ds_ev}_{sample_space}.pkl'
    else:
        if dataset == 'barrel':
            name_model = f'{path}{name}_LR_{nb_digit}_{ds_ev}_{sample_space}.pkl'
        else:
            hotshom, homeotest = netparam(name, filt, tau, nbclust, sigma, homeinv, jitter, timestr[:10], dataset, R, nb_learn = nb_learn, verbose=verbose)
            name_model = f'{path}{hotshom.get_fname()}_LR_{nb_digit}_{ds_ev}_{sample_space}.pkl'
            print(name_model)

    if isfile(name_model):
        if verbose:
            print('loading existing model')
            print(name_model)
        with open(name_model, 'rb') as file:
            logistic_model, losses = pickle.load(file)
    else:
        
        if dataset=='barrel':
            loader, train_dataset, nb_pola, time_scale = get_loader_barrel(name, 
               path, 
               True, 
               filt, 
               tau, 
               nbclust, 
               sigma, 
               homeinv, 
               jitter, 
               timestr, 
               dataset, 
               R, 
               num_workers,
               tau_cla,
               sample_space = sample_space,
               verbose = True)
        else:
            loader, train_dataset, nb_pola, time_scale = get_loader(name, path, nb_digit, True, filt, tau, nbclust, sigma, homeinv, jitter, timestr, dataset, R, num_workers, tau_cla, subset_size = subset_size, jitonic = jitonic, ds_ev = ds_ev, sample_space=sample_space, verbose = verbose)

        torch.set_default_tensor_type("torch.DoubleTensor")
        criterion = torch.nn.BCELoss(reduction="mean")
        amsgrad = True #or False gives similar results

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if verbose:
            print(f'device -> {device} - num workers -> {num_workers}')
            
        c_int = lambda n, d : ((n - 1) // d) + 1

        N = c_int(train_dataset.sensor_size[0],sample_space)*c_int(train_dataset.sensor_size[1],sample_space)*nb_pola
        n_classes = len(train_dataset.classes)
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

        pbar.close()
        with open(name_model, 'wb') as file:
            pickle.dump([logistic_model, losses], file, pickle.HIGHEST_PROTOCOL)
            
    return logistic_model, losses

def predict_data(model, 
                 name,
                 timestr,
                 path,
                 filt,
                 tau,
                 R,
                 nbclust,
                 sigma,
                 homeinv,
                 jitter,
                 dataset,
                 nb_digit,
                 ds_ev,
                 tau_cla,
                 jitonic = [None, None],
                 subset_size = None,
                 num_workers = 0,
                 sample_space=1,
                 verbose=False
        ):
    
    with torch.no_grad():
        if dataset=='barrel':
            loader, train_dataset, nb_pola, time_scale = get_loader_barrel(name, 
               path, 
               False, 
               filt, 
               tau, 
               nbclust, 
               sigma, 
               homeinv, 
               jitter, 
               timestr, 
               dataset, 
               R, 
               num_workers,
               tau_cla,
               sample_space=sample_space,
               verbose = True)
        else:
            loader, test_dataset, nb_pola, time_scale = get_loader(name, path, nb_digit, False, filt, tau, nbclust, sigma, homeinv, jitter, timestr, dataset, R, num_workers, tau_cla, subset_size = subset_size, jitonic = jitonic, ds_ev = ds_ev, sample_space=sample_space, verbose = verbose)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if verbose:
            print(f'device -> {device} - num workers -> {num_workers}')
        
        logistic_model = model.to(device)
        
        pbar = tqdm(total=nb_digit)
        likelihood, true_target, timing = [], [], []
        for X, label in loader:
            X = X.to(device)
            X, label = X.squeeze(0), label.squeeze(0)
            n_events = X.shape[0]
            #labels = label*torch.ones(n_events).type(torch.LongTensor)
            outputs = logistic_model(X)
            likelihood.append(outputs.cpu().numpy())
            #pred_target.append(torch.argmax(outputs, dim=1).cpu().numpy())
            true_target.append(label.cpu().numpy())
            timing.append(label.cpu().numpy())
            pbar.update(1)
        pbar.close()

    return likelihood, true_target, time_scale

def classification_numbevents(likelihood, true_target, thres, nb_test, chance, lenmat=30000, verbose=False):

    matscor = np.zeros([nb_test,lenmat])
    matscor[:] = np.nan
    sample = 0
    lastac = 0
    maxprobac = 0
    maxevac = 0
    number_events = []
    for likelihood_, true_target_ in zip(likelihood, true_target):
        if len(likelihood_)==0:
            sample+=1
        else:
            numbev = len(likelihood_)
            pred_target = np.zeros(numbev)
            pred_target[:] = np.nan
            score = pred_target.copy()
            number_events.append(numbev)
            pred_target = np.argmax(likelihood_, axis = 1).astype('float32')
            if thres:
                pred_target[np.max(likelihood_, axis=1)<thres]=np.nan 
            score[pred_target==true_target_] = 1
            score[np.all([pred_target!=true_target_, ~np.isnan(pred_target)], axis=0)] = 0
            matscor[sample,:numbev] = score
            if pred_target[-1]==true_target_:
                lastac+=1
            maximum_likelihood_pred = np.unravel_index(np.argmax(likelihood_), likelihood_.shape)[1]
            if maximum_likelihood_pred==true_target_:
                maxprobac+=1
            digits, counts = np.unique(pred_target, return_counts=True)
            if digits[np.argmax(counts)]==true_target_:
                maxevac+=1
            sample+=1

    meanac = np.nanmean(matscor)
    onlinac = np.nanmean(matscor, axis=0)
    lastac/=nb_test
    maxevac/=nb_test
    maxprobac/=nb_test
    truepos = len(np.where(matscor==1)[0])
    falsepos = len(np.where(matscor==0)[0])
    maxevents = np.where(np.isnan(onlinac)==0)[0][-1]
    onlinac = onlinac[:maxevents]
    zeroclassif = np.sum(np.isnan(np.nanmean(matscor,axis=1)))
    print(f'{zeroclassif} samples where not classified')
    meanac = (zeroclassif*chance + (nb_test-zeroclassif)*meanac)/nb_test
    onlinac = (zeroclassif*chance + (nb_test-zeroclassif)*onlinac)/nb_test
    lastev = np.percentile(number_events, 95)
    if verbose:
        print(f'{np.mean(meanac)=:.3f}')
        plt.plot(onlinac);
        plt.xlabel('number of events');
        plt.ylabel('online accuracy');
        plt.title('LR classification results evolution as a function of the number of events');
    
    return meanac, onlinac, lastac, maxprobac, maxevac, truepos, falsepos, lastev

def classification_timescale(likelihood, true_target, time_scale, timestep, thres, nb_test, chance, maximumtime = 1e6, verbose=False):
    matscor = np.zeros([nb_test,int(maximumtime//timestep)])
    matscor[:] = np.nan
    sample = 0
    lastac = 0
    recording_time = []
    for likelihood_, true_target_, time_scale_ in zip(likelihood, true_target, time_scale):
        if len(likelihood_)==0:
            sample+=1
        else:
            #time_scale_ -= time_scale_[0]
            pred_target = np.zeros(len(likelihood_))
            pred_target[:] = np.nan
            duration = time_scale_[-1]
            timesteps = np.linspace(0,duration,duration//timestep)
            recording_time.append(duration)
            pred_target = np.argmax(likelihood_, axis = 1).astype('float32')
            if thres:
                pred_target[np.max(likelihood_, axis=1)<thres]=np.nan 
            previous_ind = 0
            for idx, step in enumerate(timesteps):
                if time_scale_[previous_ind+1]>=step:
                    matscor[sample,idx] = np.nan
                else:
                    ind = np.where(time_scale_<=step)[0][-1]
                    digits, counts = np.unique(pred_target[previous_ind:ind], return_counts=True)
                    matscor[sample,idx] = digits[np.argmax(counts)]==true_target_
            sample+=1
            
    meanac = np.nanmean(matscor)
    onlinac = np.nanmean(matscor, axis=0)
    lastac/=nb_test
    truepos = len(np.where(matscor==1)[0])
    falsepos = len(np.where(matscor==0)[0])
    lastime = np.percentile(recording_time, 95)

    maxevents = np.where(np.isnan(onlinac)==0)[0][-1]
    onlinac = onlinac[:maxevents]
    
    zeroclassif = np.sum(np.isnan(np.nanmean(matscor,axis=1)))
    print(f'{zeroclassif} samples where not classified')
    meanac = (zeroclassif*chance + (nb_test-zeroclassif)*meanac)/nb_test
    onlinac = (zeroclassif*chance + (nb_test-zeroclassif)*onlinac)/nb_test
        
    if verbose:
        print(f'{np.mean(accuracy)=:.3f}')
        plt.plot(onlinac[:limit]/onlincount[:limit]);
        plt.xlabel('number of events');
        plt.ylabel('online accuracy');
        plt.title('LR classification results evolution as a function of the number of events');
    
    timesteps = np.linspace(0,lastime,lastime//timestep)
    
    return meanac, onlinac, timesteps, lastime, truepos, falsepos

#___________________________________________________________________________________________
#___________________________________________________________________________________________


#_______________________________TO_RUN_HOTS_________________________________________________
#___________________________________________________________________________________________

def netparam(name, filt, tau, nbclust, sigma, homeinv, jitter, timestr, dataset, R, nb_learn=10, maxevts = None, ds_ev = None, jitonic = [None,None], verbose = False):
    if verbose:
        print(f'The dataset used is: {dataset}')
    if name=='hots':
        homeo = False
        homeotest = False
        krnlinit = 'first'
        hotshom = network(krnlinit=krnlinit, filt=filt, tau=tau, R=R, nbclust=nbclust, homeo=homeo, sigma=sigma, homeinv=homeinv, jitter=jitter, timestr=timestr)
        hotshom = hotshom.learning1by1(dataset=dataset, nb_digit = nb_learn, maxevts = maxevts, ds_ev = ds_ev, jitonic = jitonic, verbose=verbose)
    elif name=='homhots':
        homeo = True
        homeotest = False
        krnlinit = 'rdn'
        hotshom = network(krnlinit=krnlinit, filt=filt, tau=tau, R=R, nbclust=nbclust, homeo=homeo, sigma=sigma, homeinv=homeinv, jitter=jitter, timestr=timestr)
        hotshom = hotshom.learningall(dataset=dataset, nb_digit = nb_learn, maxevts = maxevts, ds_ev = ds_ev, jitonic = jitonic, verbose=verbose)
    elif name=='fullhom':
        homeo = True
        homeotest = True
        krnlinit = 'rdn'
        hotshom = network(krnlinit=krnlinit, filt=filt, tau=tau, R=R, nbclust=nbclust, homeo=homeo, sigma=sigma, homeinv=homeinv, jitter=jitter, timestr=timestr)
        hotshom = hotshom.learningall(dataset=dataset, nb_digit = nb_learn, maxevts = maxevts, ds_ev = ds_ev, jitonic = jitonic, verbose=verbose)
    elif name=='onlyonline':
        homeo = False
        homeotest = False
        krnlinit = 'rdn'
        hotshom = network(krnlinit=krnlinit, filt=filt, tau=tau, R=R, nbclust=nbclust, homeo=homeo, sigma=sigma, homeinv=homeinv, jitter=jitter, timestr=timestr)
        hotshom = hotshom.learningall(dataset=dataset, nb_digit = nb_learn, maxevts = maxevts, ds_ev = ds_ev, jitonic = jitonic, verbose=verbose)
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

#___________________________________________________________________________________________
#___________________________________________________________________________________________

def plotjitter(fig, ax, jit, score, param = [0.8, 22, 4, 0.1], color='red', label='name', nb_class=10, n_epo = 33, fitting = True, logscale = False):
    score_stat = np.zeros([3,len(jit)])
    q = [0.05,0.95]
    for i in range(score.shape[1]):
        mean = np.mean(score[:,i])
        if np.unique(score[:,i]).shape[0]==1:
            score_stat[0,i], score_stat[1,i], score_stat[2,i] = mean, mean, mean
        else:
            var = np.std(score[:,i])
            alpha=mean**2*(1-mean)/var-mean
            be=alpha*(1-mean)/mean
            paramz = beta.fit(score[:,i], a=alpha, b=be, floc=0, fscale = 1)
            score_stat[0,i], score_stat[2,i] = beta.ppf(q, a=paramz[0], b=paramz[1])
            score_stat[1,i] = np.mean(score[:,i])

    if fitting:
        if logscale:
            jit_fit = np.log10(jit)
        else:
            jit_fit = jit
        logistic_model, loss, fit = fit_jitter(param, jit_fit, np.array(score_stat[1,:]), num_epochs=n_epo, verbose=False) 
        x_fit = np.arange(jit_fit[0],jit_fit[-1],(jit_fit[-1]-jit_fit[0])/100)

        if logscale:
            ax.semilogx(10**x_fit, fit.detach().numpy()*100, color=color, lw=1)
        else:
            ax.plot(x_fit, fit.detach().numpy()*100, color=color, lw=1)
    if logscale:
        ax.semilogx(jit, score_stat[1,:]*100, '.',color=color, label=label)
    else:
        ax.plot(jit, score_stat[1,:]*100, '.',color=color, label=label)
    ax.fill_between(jit, score_stat[2,:]*100, score_stat[0,:]*100, facecolor=color, edgecolor=None, alpha=.3)
        
    x = []
    if fitting:
        # semi-saturation levels
        chance = 1/nb_class
        valmax = score_stat[1,0]
        semisat = (valmax-chance)/2
        y = 1
        x = 0
        while y>semisat:
            x += 0.01
            y = logistic_model(x)

    return fig, ax, x

#___________________________FIT_____________________________________________________________
#___________________________________________________________________________________________

def fit_jitter(param,
    theta,
    y,
    learning_rate=0.005,
    batch_size=256,  # gamma=gamma,
    num_epochs=2 ** 13 + 1,
    betas=(0.9, 0.999),
    verbose=False, **kwargs
):
    
    torch.set_default_tensor_type("torch.DoubleTensor")
    criterion = torch.nn.BCELoss(reduction="mean")

    class LRModel_jitter(torch.nn.Module):
        def __init__(self, bias=True, logit0=0, jitter0=0, log_wt=torch.log(2*torch.ones(1)),  n_classes=10):
            super(LRModel_jitter, self).__init__()
            self.jitter0 = torch.nn.Parameter(jitter0 * torch.ones(1))
            self.logit0 = torch.nn.Parameter(logit0 * torch.ones(1))
            self.log_wt = torch.nn.Parameter(log_wt * torch.ones(1))
            self.n = torch.nn.Parameter(n * torch.ones(1))
            self.n_classes = n_classes

        def forward(self, jitter):
            p0 = torch.sigmoid(self.logit0)
            p = torch.sigmoid((self.jitter0-jitter)/torch.exp(self.log_wt))
            out = 1/self.n_classes + (1 - p0 - 1/self.n_classes) * p
            #out = 1-p0 / 2 + (1 - p0) * torch.sigmoid((jitter-self.jitter0 )/torch.exp(self.log_wt))
            return out
        
    class NKModel_jitter(torch.nn.Module):
        def __init__(self, bias=True, Rmax=1, jitter0=0, powa=2, low = 0.1, n_classes=10):
            super(NKModel_jitter, self).__init__()
            self.jitter0 = torch.nn.Parameter(jitter0 * torch.ones(1))
            self.Rmax = torch.nn.Parameter(Rmax * torch.ones(1))
            self.powa = torch.nn.Parameter(powa * torch.ones(1))
            self.low = torch.nn.Parameter(low * torch.ones(1))
            self.n_classes = n_classes

        def forward(self, jitter):
            x = jitter**self.powa
            semisat = self.jitter0**self.powa
            out = self.Rmax-self.Rmax*x/(x+semisat)+self.low
            return out
        
    amsgrad = True  # gives similar results

    Theta, labels = torch.Tensor(theta[:, None]), torch.Tensor(y[:, None])
    loader = DataLoader(
        TensorDataset(Theta, labels), batch_size=batch_size, shuffle=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #logistic_model = LRModel_jitter(logit0=param[0],jitter0=param[1],log_wt=torch.from_numpy(param[2]))
    logistic_model = NKModel_jitter(Rmax=param[0],jitter0=param[1], powa=param[2], low = param[3])
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
    Theta, Theta_fine, labels = torch.Tensor(theta[:, None]).to(device), torch.arange(0,theta[-1],theta[-1]/100).to(device), torch.Tensor(y[:, None]).to(device)
    outputs = logistic_model(Theta)
    loss = criterion(outputs, labels).item() / len(theta)
    
    fit = torch.squeeze(logistic_model(Theta_fine)).cpu()
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

def histoscore(trainmap,testmap, weights='distance',verbose = True):
    
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

def knn(trainmap,testmap,k, weights = 'uniform', metric = 'euclidean'):
    from sklearn.neighbors import KNeighborsClassifier

    X_train = np.array([trainmap[i][1]/np.sum(trainmap[i][1]) for i in range(len(trainmap))]).reshape(len(trainmap),len(trainmap[0][1]))
    knn = KNeighborsClassifier(n_neighbors=k, weights=weights, metric = metric)
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
    #pbar = tqdm(total=int(len(testmap)))
    for i in range(len(testmap)):
        #pbar.update(1)
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
    #pbar.close()
    return accuracy/total
