#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 17:16:50 2022

@author: alain

for EMNIST dataset
need to create the dataset using leaf 
https://github.com/TalwalkarLab/leaf/tree/master/data/femnist

"""

#%%

import numpy as np
import torch
import copy
from utils.options import args_parser
import torch.utils.data as data_utils
from torchvision import datasets, transforms
from utils.sampling import noniid
from torch.utils.data import Dataset
from torchvision.datasets import EMNIST
from torch.utils.data import DataLoader, Subset



import random

trans_mnist = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))])
trans_cifar10_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])
trans_cifar10_val = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
trans_cifar100_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                               std=[0.267, 0.256, 0.276])])
trans_cifar100_val = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                              std=[0.267, 0.256, 0.276])])

import random

def split_interval(K):
    intervals = []
    current_start = 0
    for i in range(K):
        current_end = current_start + random.uniform(0, (1 - current_start)*0.5)
        intervals.append([current_start, current_end])
        current_start = current_end
    return intervals
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs, name=None):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.name = name

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        if self.name is None:
            image, label = self.dataset[self.idxs[item]]
        elif 'femnist' in self.name:
            image = torch.reshape(torch.tensor(self.dataset['x'][item]),(1,28,28))
            label = torch.tensor(self.dataset['y'][item])
        elif 'sent140' in self.name:
            image = self.dataset['x'][item]
            label = self.dataset['y'][item]
        else:
            image, label = self.dataset[self.idxs[item]]
        return image, label

def noniid_tensor(dataset, num_users, shard_per_user, num_classes, rand_set_all=[]):
    """
    Sample non-I.I.D client data 
    
    shard per user : classes per client
    
    """
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    idxs_dict = {}
    count = 0
    for i in range(len(dataset)):
        #label = torch.tensor(dataset.targets[i]).item()
        #label = torch.tensor(dataset.tensors[1][i]).item()
        label = dataset.tensors[1].clone().detach()[i].item()
        if label < num_classes and label not in idxs_dict.keys():
            idxs_dict[label] = []
        if label < num_classes:
            idxs_dict[label].append(i)
            count += 1

    shard_per_class = int(shard_per_user * num_users / num_classes)
    samples_per_user = int( count/num_users )
    # whether to sample more test samples per user
    if (samples_per_user < 10):
        double = True
    else:
        double = False

    for label in idxs_dict.keys():
        x = idxs_dict[label]
        num_leftover = len(x) % shard_per_class
        leftover = x[-num_leftover:] if num_leftover > 0 else []
        x = np.array(x[:-num_leftover]) if num_leftover > 0 else np.array(x)
        x = x.reshape((shard_per_class, -1))
        x = list(x)

        for i, idx in enumerate(leftover):
            x[i] = np.concatenate([x[i], [idx]])
        idxs_dict[label] = x

    if len(rand_set_all) == 0:
        rand_set_all = list(range(num_classes)) * (shard_per_class)
        rand_set_all = rand_set_all[:num_users*shard_per_user]
        random.shuffle(rand_set_all)
        rand_set_all = np.array(rand_set_all).reshape((num_users, -1))

    # divide and assign
    for i in range(num_users):
        if double:
            rand_set_label = list(rand_set_all[i]) * 50
        else:
            rand_set_label = rand_set_all[i]
        rand_set = []
        for label in rand_set_label:
            #rint('*',len(idxs_dict[label]))
            idx = np.random.choice(len(idxs_dict[label]), replace=False)
            if (samples_per_user < 100):
                rand_set.append(idxs_dict[label][idx])
            else:
                rand_set.append(idxs_dict[label].pop(idx))
        dict_users[i] = np.concatenate(rand_set)

    test = []
    for key, value in dict_users.items():
        #x = np.unique(torch.tensor(dataset.tensors[1])[value])
        test.append(value)
    test = np.concatenate(test)

    return dict_users, rand_set_all

def generate_gaussians(n_class,dim,n_per_class,centers):

    X = torch.zeros((0,dim))
    std = 1
    for i in range(n_class):
        if i == 0:
            X = torch.randn(n_per_class,dim)*std + centers[i]
            y = torch.ones(n_per_class)*i
        else:
            aux = torch.randn(n_per_class,dim)*std + centers[i]
            aux_y = torch.ones(n_per_class)*i
            
            X = torch.cat((X,aux),dim=0)
            y = torch.cat((y,aux_y),dim=0)
    return X, y 

def generate_weird_gaussians(n_class,dim,n_per_class,centers):

    X = torch.zeros((0,dim))
    std = 1
    std = torch.rand(n_class,dim)*2

    for i in range(n_class):
        if i == 0:
            X = torch.randn(n_per_class,dim)@torch.diag(std[i,:]) + centers[i]

            y = torch.ones(n_per_class)*i
        else:
            #aux = torch.randn(n_per_class,dim)*std + centers[i]
            aux = torch.randn(n_per_class,dim)@torch.diag(std[i,:]) + centers[i]

            #aux_1 = torch.randn(n_per_class,dim)*std + centers[i] + 3*torch.randn_like(centers[i])
            #aux = torch.cat((aux,aux_1),dim=0)
            aux_y = torch.ones(n_per_class)*i
            
            #aux_y = torch.ones(n_per_class)*i
            X = torch.cat((X,aux),dim=0)
            y = torch.cat((y,aux_y),dim=0)
    return X, y 
        

def generate_2gaussians(n_class,dim,n_per_class,centers):

    X = torch.zeros((0,dim))
    assert(2*n_class==centers.shape[0])
    std = torch.rand(2*n_class,dim)*2 + 1
    for i in range(0,n_class):
        if i == 0:
            X = torch.randn(n_per_class//2,dim)@torch.diag(std[i,:]) + centers[i]
            X_1 = torch.randn(n_per_class//2,dim)@torch.diag(std[i+n_class,:]) + centers[i+n_class] 
            X = torch.cat((X,X_1),dim=0)
            y = torch.ones(n_per_class)*i
        else:
            aux1 = torch.randn(n_per_class//2,dim)@torch.diag(std[i,:])+ centers[i]
            aux2 = torch.randn(n_per_class//2,dim)@torch.diag(std[i+n_class,:]) + centers[i+n_class] 
            aux = torch.cat((aux1,aux2),dim=0)

            aux_y = torch.ones(n_per_class)*i
            
            X = torch.cat((X,aux),dim=0)
            y = torch.cat((y,aux_y),dim=0)
    return X, y 
def generated_perturbed_data(Xa, ya, Xt,yt, args, dim_pertub_max = 20):
        num_users = args.num_users
        shard_per_user = args.shard_per_user
        n_class = args.num_classes
        dataset_train = data_utils.TensorDataset((Xa).float(), (ya).long())
        dataset_test = data_utils.TensorDataset((Xt).float(), (yt).long())
        dict_users_train, rand_set_all = noniid_tensor(dataset_train, num_users, shard_per_user, n_class)
        dict_users_test, _ = noniid_tensor(dataset_test, num_users, shard_per_user, n_class,rand_set_all=rand_set_all)
        
        if args.subsample_client_data:
            subsample_client_data(dict_users_train,args=args)
        
        user_data  = {}
        user_data_addeddim  = {}
        user_test_data  = {}
        user_test_data_addeddim  = {}
        for user in range(num_users):
            X_aux = dataset_train.tensors[0][dict_users_train[user]]
            y_aux = dataset_train.tensors[1][dict_users_train[user]]
            if dim_pertub_max > 0:
                addeddim = int(torch.rand(1)*dim_pertub_max)+1
                if addeddim  > 0:
                    X_noise = torch.randn(X_aux.shape[0],addeddim)
                    X_aux = torch.cat((X_aux,X_noise),dim=1)
                    aux_data = data_utils.TensorDataset(X_aux, y_aux)
                else:
                    addeddim = 0
                    aux_data = data_utils.TensorDataset(X_aux, y_aux)

            else:
                addeddim = 0
                aux_data = data_utils.TensorDataset(X_aux, y_aux)
            user_data[user] = aux_data
            user_data_addeddim[user] = addeddim
            
            

            X_aux = dataset_test.tensors[0][dict_users_test[user]]
            y_aux = dataset_test.tensors[1][dict_users_test[user]]

            if addeddim  > 0:
                X_noise = torch.randn(X_aux.shape[0],addeddim)
                X_aux = torch.cat((X_aux,X_noise),dim=1)
                aux_data = data_utils.TensorDataset(X_aux, y_aux)
            else:
                addeddim = 0
                aux_data = data_utils.TensorDataset(X_aux, y_aux)


            user_test_data[user] = aux_data
            user_test_data_addeddim[user] = addeddim
        return user_data, user_data_addeddim,user_test_data,user_test_data_addeddim

def subsample_client_data(dict_users,args=None):
    for idx in range(len(dict_users)):
        index = dict_users[idx]
        if np.random.rand(1) > 0.5:
            aux = np.minimum((np.random.rand(1))*100+5,100)
        else:
            aux = np.minimum((np.random.rand(1)/5)*100+15,100) # max 25%
        percent_to_keep = int(len(index)*aux/100)
        dict_users[idx] = index[np.random.choice(index.shape[0],percent_to_keep)]
    
    



def get_toy_2(args):
    
    # Dataset in which client data varies by adding Gaussian noisy features
    
    args.dim = 5
    args.dim_pertub_max = 10
    args.n_per_class = 2000
    args.subsample_client_data = True
    args.num_classes = 20
    dim = args.dim
    n_class = args.num_classes
    n_per_class = args.n_per_class
    n_per_class_t = 1000

    
    
    centers = torch.randn(n_class,dim)*0.8
    Xa, ya = generate_gaussians(n_class, dim, n_per_class, centers) 
    Xt, yt = generate_gaussians(n_class, dim, n_per_class_t, centers)
    data_ = generated_perturbed_data(Xa, ya, Xt,yt,args,dim_pertub_max = args.dim_pertub_max)
    user_data, user_data_addeddim = data_[0], data_[1]
    user_test_data,user_test_data_addeddim = data_[2], data_[3] 
    
    return user_data, user_data_addeddim, user_test_data, user_test_data_addeddim


def get_toy_2_projection(args):
    
    # Dataset in which client data varies by adding Gaussian noisy features
    
    args.dim = 5
    args.dim_pertub_max = 10
    args.n_per_class = 2000
    args.subsample_client_data = True
    args.num_classes = 20
    dim = args.dim
    n_class = args.num_classes
    n_per_class = args.n_per_class
    n_per_class_t = 1000

    
    
    centers = torch.randn(n_class,dim)*0.8
    Xa, ya = generate_gaussians(n_class, dim, n_per_class, centers) 
    Xt, yt = generate_gaussians(n_class, dim, n_per_class_t, centers)
    data_ = generated_perturbed_data(Xa, ya, Xt,yt,args,dim_pertub_max = args.dim_pertub_max)
    user_data, user_data_addeddim = data_[0], data_[1]
    user_test_data,user_test_data_addeddim = data_[2], data_[3] 
    
        # projection on 
    from sklearn import manifold
    for idx in range(args.num_users):
        print('*',idx)
        # xa, ya = user_data[idx].tensors[0].numpy(), user_data[idx].tensors[1].numpy()
        # xt = user_test_data[idx].tensors[0].numpy()
        # X = np.concatenate((xa,xt),axis=0)
        # projection = manifold.TSNE(n_components=5, perplexity=20,method='exact',init='random')
        # Xp = projection.fit_transform(X)
        # xat,xtt = Xp[:xa.shape[0]], Xp[xa.shape[0]:]
        # user_data[idx].tensors[0] = torch.from_numpy(xa)
        # user_test_data[idx].tensors[0] = torch.from_numpy(xt)
    
        xa, ya = user_data[idx].tensors[0].numpy(), user_data[idx].tensors[1].numpy()
        xt, yt = user_test_data[idx].tensors[0].numpy(), user_test_data[idx].tensors[1].numpy()
        X = np.concatenate((xa,xt),axis=0)
        projection = manifold.TSNE(n_components=args.dim_latent, perplexity=20,init='random',n_iter=250,method='exact')
        Xp = projection.fit_transform(X)
        xat,xtt = Xp[:xa.shape[0]], Xp[xa.shape[0]:]
        user_data[idx] = data_utils.TensorDataset(torch.from_numpy(xat),torch.from_numpy(ya))
        user_test_data[idx] = data_utils.TensorDataset(torch.from_numpy(xtt),torch.from_numpy(yt))
    
    
    
    return user_data, user_data_addeddim, user_test_data, user_test_data_addeddim



def get_toy_3(args):

    # Dataset in which client data varies through a random Gaussian linear 
    # transformation    


    min_dim = 3  #min dimension of the linear transformation
    args.dim = 5
    args.dim_pertub_max = 100 # max dim of output
    args.n_per_class = 2000
    args.subsample_client_data = True
    args.num_classes = 20
    n_per_class = args.n_per_class
    num_users = args.num_users
    shard_per_user = args.shard_per_user
    n_class = args.num_classes
    
    
    centers = torch.randn(n_class,args.dim)*0.5
    Xa, ya = generate_gaussians(n_class, args.dim, n_per_class, centers) 
    Xt, yt = generate_gaussians(n_class, args.dim, n_per_class, centers)




    dataset_train = data_utils.TensorDataset((Xa).float(), (ya).long())
    dataset_test = data_utils.TensorDataset((Xt).float(), (yt).long())
    dict_users_train, rand_set_all = noniid_tensor(dataset_train, num_users, shard_per_user, n_class)
    dict_users_test, _ = noniid_tensor(dataset_test, num_users, shard_per_user, n_class,rand_set_all=rand_set_all)
    
    if args.subsample_client_data:
        subsample_client_data(dict_users_train,args=args)
    
    user_data  = {}
    user_data_addeddim  = {}
    user_test_data  = {}
    user_test_data_addeddim  = {}
    for user in range(num_users):
        
        X_aux = dataset_train.tensors[0][dict_users_train[user]]
        y_aux = dataset_train.tensors[1][dict_users_train[user]]
        dim_aux = torch.randint(min_dim,args.dim_pertub_max,(1,))
        L = torch.randn((args.dim,dim_aux))
        aux_data = data_utils.TensorDataset(X_aux@L, y_aux)
            
        user_data[user] = aux_data
        user_data_addeddim[user] = dim_aux
            
        
        X_aux = dataset_test.tensors[0][dict_users_test[user]]
        y_aux = dataset_test.tensors[1][dict_users_test[user]]

        aux_data = data_utils.TensorDataset(X_aux@L, y_aux)
            
        user_test_data[user] = aux_data
        user_test_data_addeddim[user] = dim_aux

    return user_data, user_data_addeddim,user_test_data,user_test_data_addeddim

def get_toy_3_projection(args):

    # Dataset in which client data varies through a random Gaussian linear 
    # transformation    


    min_dim = 3  #min dimension of the linear transformation
    args.dim = 5
    args.dim_pertub_max = 100 # max dim of output
    args.n_per_class = 2000
    args.subsample_client_data = True
    args.num_classes = 20
    n_per_class = args.n_per_class
    num_users = args.num_users
    shard_per_user = args.shard_per_user
    n_class = args.num_classes
    
    
    centers = torch.randn(n_class,args.dim)*0.5
    Xa, ya = generate_gaussians(n_class, args.dim, n_per_class, centers) 
    Xt, yt = generate_gaussians(n_class, args.dim, n_per_class, centers)




    dataset_train = data_utils.TensorDataset((Xa).float(), (ya).long())
    dataset_test = data_utils.TensorDataset((Xt).float(), (yt).long())
    dict_users_train, rand_set_all = noniid_tensor(dataset_train, num_users, shard_per_user, n_class)
    dict_users_test, _ = noniid_tensor(dataset_test, num_users, shard_per_user, n_class,rand_set_all=rand_set_all)
    
    if args.subsample_client_data:
        subsample_client_data(dict_users_train,args=args)
    
    user_data  = {}
    user_data_addeddim  = {}
    user_test_data  = {}
    user_test_data_addeddim  = {}
    for user in range(num_users):
        
        X_aux = dataset_train.tensors[0][dict_users_train[user]]
        y_aux = dataset_train.tensors[1][dict_users_train[user]]
        dim_aux = torch.randint(min_dim,args.dim_pertub_max,(1,))
        L = torch.randn((args.dim,dim_aux))
        aux_data = data_utils.TensorDataset(X_aux@L, y_aux)
            
        user_data[user] = aux_data
        user_data_addeddim[user] = dim_aux
            
        
        X_aux = dataset_test.tensors[0][dict_users_test[user]]
        y_aux = dataset_test.tensors[1][dict_users_test[user]]

        aux_data = data_utils.TensorDataset(X_aux@L, y_aux)
            
        user_test_data[user] = aux_data
        user_test_data_addeddim[user] = dim_aux

    from sklearn import manifold
    for idx in range(args.num_users):
        print('*',idx)

        xa, ya = user_data[idx].tensors[0].numpy(), user_data[idx].tensors[1].numpy()
        xt, yt = user_test_data[idx].tensors[0].numpy(), user_test_data[idx].tensors[1].numpy()
        X = np.concatenate((xa,xt),axis=0)
        projection = manifold.TSNE(n_components=args.dim_latent, perplexity=20,init='random',n_iter=250,method='exact')
        Xp = projection.fit_transform(X)
        xat,xtt = Xp[:xa.shape[0]], Xp[xa.shape[0]:]
        user_data[idx] = data_utils.TensorDataset(torch.from_numpy(xat),torch.from_numpy(ya))
        user_test_data[idx] = data_utils.TensorDataset(torch.from_numpy(xtt),torch.from_numpy(yt))
    
    
    
    return user_data, user_data_addeddim,user_test_data,user_test_data_addeddim




def get_mnist_usps(args):
    
    if not hasattr(args,'image_size'):
        pre_process = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(
                                            mean=(0.5,),
                                            std =(0.5,))])
    else:
        pre_process = transforms.Compose([transforms.Resize(args.image_size),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    mean=[0.5],
                                        std =[0.5])])
    if not hasattr(args,'image_size'):
        mnistpre_process = transforms.Compose([#transforms.Resize(image_size),
                                        transforms.Grayscale(1),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            mean=[0.5],
                                            std =[0.5])])
    else:
        mnistpre_process = transforms.Compose([transforms.Resize(args.image_size),
                                        transforms.Grayscale(1),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            mean=[0.5],
                                            std =[0.5])])
    usps_dataset_train = datasets.USPS(root= '../dann_dataset',
                        train=True,
                        transform=pre_process,
                        download=True)
    
    usps_dataset_test = datasets.USPS(root= '../dann_dataset',
                        train=False,
                        transform=pre_process,
                        download=True)
    
    
    mnist_dataset_train = datasets.MNIST(root='../dann_dataset/',
                                   train=True,
                                   transform=mnistpre_process,
                                   download=True)
    
    mnist_dataset_test = datasets.MNIST(root='../dann_dataset/',
                                   train=False,
                                   transform=mnistpre_process,
                                   download=True)
    

    
    usps_dict_users_train, rand_set_all = noniid(usps_dataset_train, args.num_users, args.shard_per_user, args.num_classes)
    usps_dict_users_test, rand_set_all = noniid(usps_dataset_test, args.num_users, args.shard_per_user, args.num_classes, rand_set_all=rand_set_all)

    
    mnist_dict_users_train, rand_set_all = noniid(mnist_dataset_train, args.num_users, args.shard_per_user, args.num_classes)
    mnist_dict_users_test, rand_set_all = noniid(mnist_dataset_test, args.num_users, args.shard_per_user, args.num_classes, rand_set_all=rand_set_all)

    #from Het_data import DatasetSplit
    assert(args.num_users%2 == 0)
    num_users_data = args.num_users//2
    
    ind = torch.randperm(args.num_users)

    user_data  = {}
    user_test_data  = {}
    user_data_addeddim  = {}

    for i in range(args.num_users):
        idx = ind[i]
        if idx < num_users_data:
            user_data[i]  = DatasetSplit(usps_dataset_train, usps_dict_users_train[idx.item()])
            user_test_data[i]  = DatasetSplit(usps_dataset_test, usps_dict_users_test[idx.item()])
            user_data_addeddim[i] = 'usps'
        else:
            idx = idx - num_users_data
            user_data[i]  = DatasetSplit(mnist_dataset_train, mnist_dict_users_train[idx.item()])
            user_test_data[i]  = DatasetSplit(mnist_dataset_test, mnist_dict_users_test[idx.item()])
            user_data_addeddim[i] = 'mnist'

    return user_data, user_data_addeddim, user_test_data, user_data_addeddim


def get_textcaps(args):

    import numpy as np
    
    data = torch.load('tensors.pt')
    y = np.array(data['y'])
    if 1:
        if args.num_classes == 10:
            list_keep = [0,1,2,3,4,5,6,7,8,9]
            ratio_train = 0.7
        else:
            list_keep = [0,1,2,3]
            ratio_train = 0.8
        
        for i in list_keep:
            ind = np.where(y==i)[0]
            if i == 0:
                ind_to_keep=ind
            else:
                ind_to_keep = np.concatenate((ind_to_keep,ind))
    else:
        ind_to_keep = np.arange(len(y))
    
    X_im = data['X_image'][ind_to_keep,:]
    X_t = data['X_text'][ind_to_keep,:]
    y = np.array(data['y'])[ind_to_keep]
    n_samples = X_im.shape[0]
    n = int(n_samples*ratio_train)
    ind = torch.randperm(n_samples)
    X_im_train = X_im[ind[:n],:]
    X_im_test = X_im[ind[n:],:]
    X_t_train = X_t[ind[:n],:]
    X_t_test = X_t[ind[n:],:]
    y_train = torch.Tensor(y[ind[:n]])
    y_test = torch.Tensor(y[ind[n:]])
    
    im_dataset_train = data_utils.TensorDataset((X_im_train).float(), (y_train).long())
    im_dataset_test = data_utils.TensorDataset((X_im_test).float(), (y_test).long())
    
    text_dataset_train = data_utils.TensorDataset((X_t_train).float(), (y_train).long())
    text_dataset_test = data_utils.TensorDataset((X_t_test).float(), (y_test).long())
    
    
    im_dict_users_train, rand_set_all = noniid_tensor(im_dataset_train, args.num_users, args.shard_per_user, args.num_classes)
    im_dict_users_test, rand_set_all = noniid_tensor(im_dataset_test, args.num_users, args.shard_per_user, args.num_classes, rand_set_all=rand_set_all)

    
    text_dict_users_train, rand_set_all = noniid_tensor(text_dataset_train, args.num_users, args.shard_per_user, args.num_classes)
    text_dict_users_test, rand_set_all = noniid_tensor(text_dataset_test, args.num_users, args.shard_per_user, args.num_classes, rand_set_all=rand_set_all)

    #from Het_data import DatasetSplit
    assert(args.num_users%2 == 0)
    num_users_data = args.num_users//2
    
    ind = torch.randperm(args.num_users)

    user_data  = {}
    user_test_data  = {}
    user_data_addeddim  = {}

    for i in range(args.num_users):
        idx = ind[i].item()
        if idx < num_users_data:

            ind_full = 768
            ind_mini = 730
            ind_f = np.random.permutation(ind_full)
            tokeep = ind_mini + np.random.randint(ind_full-ind_mini +1)
            ind_f = ind_f[:tokeep]
                
            X_aux = X_t_train[text_dict_users_train[idx]]
            X_aux = X_aux[:,ind_f]
            y_aux = y_train[text_dict_users_train[idx]].long()
            aux_data = data_utils.TensorDataset(X_aux, y_aux)
            
            X_aux = X_t_test[text_dict_users_test[idx]]
            X_aux = X_aux[:,ind_f]

            y_aux = y_test[text_dict_users_test[idx]].long()
            aux_test_data = data_utils.TensorDataset(X_aux, y_aux)
            user_data[i] = aux_data
            user_test_data[i] = aux_test_data
            user_data_addeddim[i] = tokeep
        else:
            idx = idx - num_users_data
            ind_full = 512
            ind_mini = 480
            ind_f = np.random.permutation(ind_full)
            tokeep = ind_mini + np.random.randint(ind_full-ind_mini +1)
            ind_f = ind_f[:tokeep]
            
            X_aux = X_im_train[im_dict_users_train[idx]]
            X_aux = X_aux[:,ind_f]

            y_aux = y_train[im_dict_users_train[idx]].long()
            aux_data = data_utils.TensorDataset(X_aux, y_aux)
            
            X_aux = X_im_test[im_dict_users_test[idx]]
            X_aux = X_aux[:,ind_f]

            y_aux = y_test[im_dict_users_test[idx]].long()
            aux_test_data = data_utils.TensorDataset(X_aux, y_aux)
            user_data[i] = aux_data
            user_test_data[i] = aux_test_data            
            user_data_addeddim[i] = tokeep



    return user_data, user_data_addeddim, user_test_data, user_data_addeddim

def get_reg_toy_1(args):

    dim = args.dim 
    s = torch.randn(dim)/dim
    n_test = 1000
    user_data  = {}
    user_test_data  = {}
    user_data_addeddim  = {}
    for i in range(args.num_users):
        n = (torch.randint(170,(1,)) + 20).item()

        X_ = torch.randn(n, dim)
        X_ /= torch.linalg.norm(X_, axis=0)
        A = torch.randn(dim,dim) + 1e-8
        X = copy.deepcopy(X_)@A@A
        y = X@s + torch.randn(n)*0.1
        dataset_train = data_utils.TensorDataset(X.float(),y.float())
        user_data[i]  =  dataset_train
        X_ = torch.randn(n_test, dim)
        X_ /= torch.linalg.norm(X_, axis=0)

        Xt = copy.deepcopy(X_)@A@A
        yt = Xt@s+ torch.randn(n_test)*0.1
        dataset_test = data_utils.TensorDataset(Xt.float(),yt.float())
        user_test_data[i]  =  dataset_test

    return user_data, user_data_addeddim, user_test_data, user_data_addeddim

def get_bci_full(args):
    data2 = np.load('./data/bci_users54_5c.npz', allow_pickle=True)
    dict2 = {key: data2[key] for key in data2.keys() if 'arr_' not in key}
    user = dict2['user'].item()
    meta_info = dict2['meta_info'].item()

    user_data  = {}
    user_test_data  = {}
    user_data_addeddim  = {}
    for i in range(len(user)):
        X_train, y_train = torch.from_numpy(user[i][0]), torch.from_numpy(user[i][1])
        dataset_train = data_utils.TensorDataset(X_train.float(),y_train.long())
        user_data_addeddim[i]=X_train.shape[1]
        user_data[i]  =  dataset_train
        
        
        X_test, y_test = torch.from_numpy(user[i][2]), torch.from_numpy(user[i][3])
        dataset_test = data_utils.TensorDataset(X_test.float(),y_test.long())
        user_test_data[i]  =  dataset_test
    return user_data, user_data_addeddim, user_test_data, user_data_addeddim

def get_bci_subset_specific(args):
    data2 = np.load('./data/bci_users40_5c_specific.npz', allow_pickle=True)
    dict2 = {key: data2[key] for key in data2.keys() if 'arr_' not in key}
    user = dict2['user'].item()
    meta_info = dict2['meta_info'].item()

    user_data  = {}
    user_test_data  = {}
    user_data_addeddim  = {}
    for i in range(len(user)):
        X_train, y_train = torch.from_numpy(user[i][0]), torch.from_numpy(user[i][1])
        dataset_train = data_utils.TensorDataset(X_train.float(),y_train.long())
        user_data_addeddim[i]=X_train.shape[1]
        user_data[i]  =  dataset_train
        
        
        X_test, y_test = torch.from_numpy(user[i][2]), torch.from_numpy(user[i][3])
        dataset_test = data_utils.TensorDataset(X_test.float(),y_test.long())
        user_test_data[i]  =  dataset_test
    return user_data, user_data_addeddim, user_test_data, user_data_addeddim

def get_bci_subset_common(args):
    data2 = np.load('./data/bci_users40_5c_common.npz', allow_pickle=True)
    dict2 = {key: data2[key] for key in data2.keys() if 'arr_' not in key}
    user = dict2['user'].item()
    meta_info = dict2['meta_info'].item()

    user_data  = {}
    user_test_data  = {}
    user_data_addeddim  = {}
    for i in range(len(user)):
        X_train, y_train = torch.from_numpy(user[i][0]), torch.from_numpy(user[i][1])
        dataset_train = data_utils.TensorDataset(X_train.float(),y_train.long())
        user_data_addeddim[i]=X_train.shape[1]
        user_data[i]  =  dataset_train
        
        
        X_test, y_test = torch.from_numpy(user[i][2]), torch.from_numpy(user[i][3])
        dataset_test = data_utils.TensorDataset(X_test.float(),y_test.long())
        user_test_data[i]  =  dataset_test
    return user_data, user_data_addeddim, user_test_data, user_data_addeddim


def get_textcaps_clip(args):
    import numpy as np
    
    data = torch.load('tensors_clip.pt')
    y = np.array(data['y'])
    if 1:
        if args.num_classes == 10:
            list_keep = [0,1,2,3,4,5,6,7,8,9]
            ratio_train = 0.7
        else:
            list_keep = [0,1,2,3]
            ratio_train = 0.8
        for i in list_keep:
            ind = np.where(y==i)[0]
            if i == 0:
                ind_to_keep=ind
            else:
                ind_to_keep = np.concatenate((ind_to_keep,ind))
    else:
        ind_to_keep = np.arange(len(y))
    
    print(len(ind_to_keep))
    X_im = data['X_image'][ind_to_keep,:]
    X_t = data['X_text'][ind_to_keep,:]
    y = np.array(data['y'])[ind_to_keep]
    X_im_clip = data['X_image_clip'][ind_to_keep,:]
    X_t_clip = data['X_text_clip'][ind_to_keep,:]
    print(len(y))
    n_samples = X_im_clip.shape[0]
    n = int(n_samples*ratio_train)

    ind = torch.randperm(n_samples)
    X_im_train = X_im_clip[ind[:n],:]
    X_im_test = X_im_clip[ind[n:],:]
    X_t_train = X_t_clip[ind[:n],:]
    X_t_test = X_t_clip[ind[n:],:]
    y_train = torch.Tensor(y[ind[:n]])
    y_test = torch.Tensor(y[ind[n:]])

    im_dataset_train = data_utils.TensorDataset((X_im_train).float(), (y_train).long())
    im_dataset_test = data_utils.TensorDataset((X_im_test).float(), (y_test).long())
    
    text_dataset_train = data_utils.TensorDataset((X_t_train).float(), (y_train).long())
    text_dataset_test = data_utils.TensorDataset((X_t_test).float(), (y_test).long())
    

    im_dict_users_train, rand_set_all = noniid_tensor(im_dataset_train, args.num_users, args.shard_per_user, args.num_classes)
    im_dict_users_test, rand_set_all = noniid_tensor(im_dataset_test, args.num_users, args.shard_per_user, args.num_classes, rand_set_all=rand_set_all)

    
    text_dict_users_train, rand_set_all = noniid_tensor(text_dataset_train, args.num_users, args.shard_per_user, args.num_classes)
    text_dict_users_test, rand_set_all = noniid_tensor(text_dataset_test, args.num_users, args.shard_per_user, args.num_classes, rand_set_all=rand_set_all)

    #from Het_data import DatasetSplit
    assert(args.num_users%2 == 0)
    num_users_data = args.num_users//2
    
    ind = torch.randperm(args.num_users)

    user_data  = {}
    user_test_data  = {}
    user_data_addeddim  = {}
    tokeep = 512
    for i in range(args.num_users):
        idx = ind[i].item()
        if idx < num_users_data:

            X_aux = X_t_train[text_dict_users_train[idx]]
            y_aux = y_train[text_dict_users_train[idx]].long()
            aux_data = data_utils.TensorDataset(X_aux, y_aux)
            
            X_aux = X_t_test[text_dict_users_test[idx]]
            y_aux = y_test[text_dict_users_test[idx]].long()
            aux_test_data = data_utils.TensorDataset(X_aux, y_aux)

            user_data[i] = aux_data
            user_test_data[i] = aux_test_data
            user_data_addeddim[i] = tokeep
        else:
            idx = idx - num_users_data

            
            X_aux = X_im_train[im_dict_users_train[idx]]

            y_aux = y_train[im_dict_users_train[idx]].long()
            aux_data = data_utils.TensorDataset(X_aux, y_aux)
            
            X_aux = X_im_test[im_dict_users_test[idx]]

            y_aux = y_test[im_dict_users_test[idx]].long()
            aux_test_data = data_utils.TensorDataset(X_aux, y_aux)
            user_data[i] = aux_data
            user_test_data[i] = aux_test_data            
            user_data_addeddim[i] = tokeep
    return user_data, user_data_addeddim, user_test_data, user_data_addeddim


def get_femnist(args):
        from utils.train_utils import read_data
        train_path = './data/femnist/train/'
        test_path = './data/femnist/test/'
        clients, groups, dataset_train, dataset_test = read_data(train_path, test_path)
        keep_label = [0,1,2,3,4,5,6,7,8,9]
        to_del = []
        for i in dataset_train.keys():
            data = dataset_train[i]['x']

            label = dataset_train[i]['y']
            new_data = []
            new_label = []
            for j,x in enumerate(label):
                if x in keep_label:
                    new_label.append(x) 
                    new_data.append(data[j])
            if len(new_label) > 0:   
                dataset_train[i]['x'] = new_data
                dataset_train[i]['y'] = new_label
                data = dataset_test[i]['x']
                label = dataset_test[i]['y']
                new_data = []
                new_label = []
                for j,x in enumerate(label):
                    if x in keep_label:
                        new_label.append(x) 
                        new_data.append(data[j])   
                dataset_test[i]['x'] = new_data
                dataset_test[i]['y'] = new_label
                if len(new_label) == 0:
                    to_del.append(i)
            else:
                to_del.append(i)
        for i in to_del:
            del dataset_train[i]
            del dataset_test[i]
            del clients[clients.index(i)]

        
        user_data  = {}
        user_test_data  = {}
        user_data_addeddim  = {}
        for i in range(len(clients)):
            x = torch.from_numpy(np.array(dataset_train[clients[i]]['x']))/10
            y = torch.from_numpy(np.array(dataset_train[clients[i]]['y']))
            dataset_ = data_utils.TensorDataset(x.float(),y.long())
            user_data_addeddim[i]=x.shape[1]
            user_data[i]  =  dataset_


            x = torch.from_numpy(np.array(dataset_test[clients[i]]['x']))/10
            y = torch.from_numpy(np.array(dataset_test[clients[i]]['y']))
            dataset_ = data_utils.TensorDataset(x.float(),y.long())
            user_test_data[i]  =  dataset_

        return user_data, user_data_addeddim, user_test_data, user_data_addeddim

if __name__ == '__main__':
    import sys
    sys.argv=['']
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--alg', type=str, default='FLic', help="Algorithm")
    parser.add_argument('--dataset', type=str, default='toy_2', help="choice of the dataset")

    parser.add_argument('--num_users', type=int, default=60, help="number of users")
    parser.add_argument('--shard_per_user', type=int, default=2, help="number of users")
    parser.add_argument('--num_classes', type=int, default=4, help="number of users")

    if 0:
        #%--------------------------------------------------------------------
        # reading the bci data
        #-----------------------------------------------------------------------
        import numpy as np
        data2 = np.load('bci_small.npz', allow_pickle=True)
        dict2 = {key: data2[key] for key in data2.keys() if 'arr_' not in key}
        user = dict2['user'].item()
        meta_info = dict2['meta_info'].item()

        user_data  = {}
        user_test_data  = {}
        user_data_addeddim  = {}
        for i in range(len(user)):
            X_train, y_train = torch.from_numpy(user[i][0]), torch.from_numpy(user[i][1])
            dataset_train = data_utils.TensorDataset(X_train.float(),y_train.long())
            user_data_addeddim[i]=X_train.shape[1]
            user_data[i]  =  dataset_train
            
            
            X_test, y_test = torch.from_numpy(user[i][2]), torch.from_numpy(user[i][3])
            dataset_test = data_utils.TensorDataset(X_test.float(),y_test.long())
            user_test_data[i]  =  dataset_test



    elif 0:
        # ----------------------²----------------------------------------------
        # reading and plotting the data in 2d
        #-----------------------------------------------------------------------
        args = parser.parse_args()

        args.num_users  = 100
        args.subsample_client_data = False

        data_ = get_toy_2(args)  

        user_data, user_data_addeddim = data_[0], data_[1]
        user_test_data,user_test_data_addeddim = data_[2], data_[3] 
        n_tot = 0
        n_data = []
        for idx in range(args.num_users):
            n = user_test_data[idx].tensors[0].shape[0]
            d = user_test_data[idx].tensors[0].shape[1] 
            n_tot += n
            print(idx,n,d)
            n_data.append(n)
        print(n_tot)
        
        # projection on 
        from sklearn import manifold
        for idx in range(args.num_users):
            print('*',idx)
            xa, ya = user_data[idx].tensors[0].numpy(), user_data[idx].tensors[1].numpy()
            xt, yt = user_test_data[idx].tensors[0].numpy(), user_test_data[idx].tensors[1].numpy()
            X = np.concatenate((xa,xt),axis=0)
            projection = manifold.TSNE(n_components=64, perplexity=20,init='random',n_iter=250,method='exact')
            Xp = projection.fit_transform(X)
            xat,xtt = Xp[:xa.shape[0]], Xp[xa.shape[0]:]
            user_data[idx] = data_utils.TensorDataset(torch.from_numpy(xat),torch.from_numpy(ya))
            user_test_data[idx] = data_utils.TensorDataset(torch.from_numpy(xtt),torch.from_numpy(yt))
            #user_test_data[idx].tensors[0] = torch.from_numpy(xt)
            
            
    elif 0:
        args = parser.parse_args()

        args.num_users  = 10
        args.subsample_client_data = False

        data_ = get_mnist_usps(args)

        
        user_data, user_data_addeddim = data_[0], data_[1]
    elif 1:
        args = parser.parse_args()

        args.num_users  = 10
        args.subsample_client_data = False
        from utils.train_utils import get_data, get_model, read_data
        train_path = '../leaf-master/data/femnist/data/train'
        test_path = '../leaf-master/data/femnist/data/test'
        clients, groups, dataset_train, dataset_test = read_data(train_path, test_path)
        keep_label = [0,1,2,3,4,5,6,7,8,9]
        to_del = []
        for i in dataset_train.keys():
            data = dataset_train[i]['x']

            label = dataset_train[i]['y']
            new_data = []
            new_label = []
            for j,x in enumerate(label):
                if x in keep_label:
                    new_label.append(x) 
                    new_data.append(data[j])
            if len(new_label) > 0:   
                dataset_train[i]['x'] = new_data
                dataset_train[i]['y'] = new_label
                data = dataset_test[i]['x']
                label = dataset_test[i]['y']
                new_data = []
                new_label = []
                for j,x in enumerate(label):
                    if x in keep_label:
                        new_label.append(x) 
                        new_data.append(data[j])   
                dataset_test[i]['x'] = new_data
                dataset_test[i]['y'] = new_label
                if len(new_label) == 0:
                    to_del.append(i)
            else:
                to_del.append(i)
        for i in to_del:
            del dataset_train[i]
            del dataset_test[i]
            del clients[clients.index(i)]

        args.num_users = len(clients)
        # Calculate the average
        total_sum = np.zeros((784))
        count = 0

        for d in dataset_train.values():
            aux = np.array(d['x'])
            total_sum += aux.sum(axis=0)
            count += len(d['x'])

        average = total_sum / count

        # Normalize the values
        for d in dataset_train.values():
            aux = np.array(d['x'])

            d['x'] = aux - average
        for d in dataset_test.values():
            aux = np.array(d['x'])

            d['x'] = aux - average
        
        
        user_data  = {}
        user_test_data  = {}
        user_data_addeddim  = {}
        for i in range(len(clients)):
            x = torch.from_numpy(np.array(dataset_train[clients[i]]['x']))
            y = torch.from_numpy(np.array(dataset_train[clients[i]]['y']))
            dataset_ = data_utils.TensorDataset(x.float(),y.long())
            user_data_addeddim[i]=x.shape[1]
            user_data[i]  =  dataset_


            x = torch.from_numpy(np.array(dataset_test[clients[i]]['x']))
            y = torch.from_numpy(np.array(dataset_test[clients[i]]['y']))
            dataset_ = data_utils.TensorDataset(x.float(),y.long())
            user_test_data[i]  =  dataset_

    
    #%
    
    #from utils.train_utils import get_data, get_model, read_data
    
    #dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args)
    # #%%
    # N = 100
    # n = 0
    # for idx in range(N):
    #     n += (dict_users_test[idx].shape[0])
    # print(n)
    
    
    # n_class = 10
    # dim = 2
    # n_per_class = 1000
    # num_users = 10
    # shard_per_user = 2
    
    # centers = torch.randn(2*n_class,dim)*3
    
    # X,y = generate_2gaussians(n_class, dim, n_per_class, centers)
    
    
    # Xa, ya = generate_data(n_class, dim, n_per_class, centers) 
    # Xt, yt = generate_data(n_class, dim, n_per_class, centers)
    # dataset_train = data_utils.TensorDataset((Xa).float(), (ya).long())
    # dataset_test = data_utils.TensorDataset((Xt).float(), (yt).long())
    
    # dict_users_train, rand_set_all = noniid_tensor(dataset_train, num_users, shard_per_user, n_class)
    # dict_users_test, rand_set_all = noniid_tensor(dataset_test, num_users, shard_per_user, n_class,rand_set_all=rand_set_all)
    
    # n = 0
    # for idx in range(num_users):
    #     n += (dict_users_train[idx].shape[0])
    # print(n)


    #%%
    
    # dataset_train = datasets.CIFAR10('data/cifar10', train=True, download=True, transform=trans_cifar10_train)
    # dataset_test = datasets.CIFAR10('data/cifar10', train=False, download=True, transform=trans_cifar10_val)
    # dict_users_train, rand_set_all = noniid(dataset_train, args.num_users, args.shard_per_user, args.num_classes)
    # dict_users_test, rand_set_all = noniid(dataset_test, args.num_users, args.shard_per_user, args.num_classes, rand_set_all=rand_set_all)
    # user_data  = {}
    # for user in range(args.num_users):
    #     user_data[user]  = DatasetSplit(dataset_train, dict_users_train[user])
        
    #%%
    
    
    #%%
    # Affine mapping transformation 
    # seed = args.seed
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # random.seed(seed)
    # args.dim = 5
    # args.dim_pertub_max = 10
    # args.n_per_class = 1000
    # args.subsample_client_data = True
    # args.num_classes = 10
    # from Het_data import generate_gaussians, generated_perturbed_data
    # dim = args.dim
                
    # data_ = get_toy_2(args)
    # user_data = data_[0]
    # print(user_data[0].tensors)
    # ind_full = 768
    # ind_mini = 765
    # ind_f = np.random.permutation(768)
    # tokeep = ind_mini + np.random.randint(ind_full-ind_mini +1)
    # ind_f = ind_f[:tokeep]
    # print(tokeep)
    
    #ata = torch.load('tensors.pt')
    
    
    
    #%%--------------------------------------------------------------------
    #  Testing the  Regression problem
    #----------------------------------------------------------------------------
    # if 0:
    #     import matplotlib.pyplot as plt
    #     args.num_users  = 20
    #     args.dim = 5
    #     args.device = 'cpu'
    
    #     dim = args.dim
    
        
    #     user_data, user_data_addeddim, user_test_data, user_data_addeddim = get_reg_toy_1(args)
    
    #     idx = 0
    #     from torch.utils.data import DataLoader
    
    #     data_adapt = DataLoader(dataset=user_data[idx], batch_size=100, shuffle=True
    #                                 ,drop_last=True)
        
        
    #     from Het_Nets import MLPReg_HetNN
        
    #     net = MLPReg_HetNN(5,100,1)
    #     optimizer_preproc = torch.optim.Adam(net.parameters(),lr=0.001)
    #     mse_loss = torch.nn.MSELoss()
    #     for itm in range(200):
    
    
    #         loss_tot = 0
    #         for it, (data,label) in enumerate((data_adapt)):
    #             optimizer_preproc.zero_grad()
    #             net.zero_grad()
    #             output,_ = net(data)
    #             loss = mse_loss(label,output.squeeze())
    #             loss.backward()
    #             optimizer_preproc.step()
    
    #             loss_tot +=loss.item()
    #         print(itm,loss_tot)
            
    #     #print(user_data[0].tensors[0]@s - user_data[0].tensors[1])
    #     y_pred, _ = net(user_data[0].tensors[0])
    #     error = y_pred.squeeze() -   user_data[0].tensors[1]
    #     print(torch.norm(error,p=2)/error.shape[0])
    
    #     y_pred, _ = net(user_test_data[0].tensors[0])
    #     error = y_pred.squeeze() -   user_test_data[0].tensors[1]
    #     print('test',torch.norm(error,p=2)**2/error.shape[0])
    
    
    #     XX = user_data[0].tensors[0]
    #     yy = user_data[0].tensors[1]
        
    #     shat = torch.linalg.solve(XX.T@XX + 1e-4*torch.ones(dim),XX.T@yy)    
        
    #     from HetReg_Update import hetreg_competitor_test_img_local
        
    #     a = hetreg_competitor_test_img_local(net, user_test_data[idx], args) 
    #     print(a)

    





    