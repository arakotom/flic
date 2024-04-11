#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 14:34:22 2022

@author: alain
"""

import torch
from torch import nn
import torch.nn.functional as F
import itertools



def get_model(args):
    if 'toy' in args.dataset and '2-hlayers' in args.model_type:
        net_glob = MLP_tensor(dim_in=args.dim_latent, dim_hidden=32, dim_out=args.num_classes).to(args.device)
        # specify the representation parameters (in w_glob_keys) and head parameters (all others)
        w_glob_keys = [net_glob.weight_keys[i] for i in [0,1,2]]
        w_glob_keys = list(itertools.chain.from_iterable(w_glob_keys))

    elif 'toy' in args.dataset and 'no-hlayers' in args.model_type:
        # a simpler model with one global layer and the output classification layer
        net_glob = MLP_tensor_simple(dim_in=args.dim_latent, dim_hidden=100, dim_out=args.num_classes).to(args.device)
        w_glob_keys = [net_glob.weight_keys[i] for i in [0]]
        w_glob_keys = list(itertools.chain.from_iterable(w_glob_keys))
    elif 'toy' in args.dataset and 'classif' in args.model_type:
        # no global model just the output classification layer
        # with input size is the size of latent space after proj
        net_glob = MLP_tensor_out(dim_in=args.dim_latent, dim_out=args.num_classes).to(args.device)
        w_glob_keys = []
        w_glob_keys = list(itertools.chain.from_iterable(w_glob_keys))
    elif 'toy' in args.dataset and 'average' in args.model_type:
        # 
        # 
        net_glob = MLP_tensor_simple(dim_in=args.dim_latent, dim_hidden=100, dim_out=args.num_classes).to(args.device)
        w_glob_keys = [net_glob.weight_keys[i] for i in [0,1]]
        w_glob_keys = list(itertools.chain.from_iterable(w_glob_keys)) 
        
    elif 'MU'== args.dataset and args.model_type=='classif':
        net_glob =DigitsClassifier(args.dim_latent,args.num_classes).to(args.device)
        w_glob_keys = []
        w_glob_keys = list(itertools.chain.from_iterable(w_glob_keys))
    elif 'MU'== args.dataset and args.model_type=='no-hlayers':
         net_glob = MLP_tensor_simple(dim_in=args.dim_latent, dim_hidden=100, dim_out=args.num_classes).to(args.device)
         w_glob_keys = [net_glob.weight_keys[i] for i in [0]]
         w_glob_keys = list(itertools.chain.from_iterable(w_glob_keys))
    elif 'MU-resize'== args.dataset and args.model_type=='classif':
        net_glob =DigitsClassifier(args.dim_latent,args.num_classes).to(args.device)
        w_glob_keys = []
        w_glob_keys = list(itertools.chain.from_iterable(w_glob_keys))
    elif 'MU-resize'== args.dataset and args.model_type=='no-hlayers':
         net_glob = MLP_tensor_simple(dim_in=args.dim_latent, dim_hidden=100, dim_out=args.num_classes).to(args.device)
         w_glob_keys = [net_glob.weight_keys[i] for i in [0]]
         w_glob_keys = list(itertools.chain.from_iterable(w_glob_keys))
    elif 'femnist' == args.dataset and 'classif' in args.model_type:
        net_glob =DigitsClassifier(args.dim_latent,args.num_classes).to(args.device)
        w_glob_keys = []
        w_glob_keys = list(itertools.chain.from_iterable(w_glob_keys))
    elif 'femnist'== args.dataset and args.model_type=='no-hlayers':
         net_glob = MLP_tensor_simple(dim_in=args.dim_latent, dim_hidden=100, dim_out=args.num_classes).to(args.device)
         w_glob_keys = [net_glob.weight_keys[i] for i in [0]]
         w_glob_keys = list(itertools.chain.from_iterable(w_glob_keys))
    elif 'textcaps' in args.dataset and 'classif' in args.model_type:
        

        net_glob = MLP_tensor_out(dim_in=args.dim_latent, dim_out=args.num_classes).to(args.device)
        w_glob_keys = []
        w_glob_keys = list(itertools.chain.from_iterable(w_glob_keys))
        
    elif 'textcaps' in args.dataset and 'no-hlayers' in args.model_type:
        # a simpler model with one global layer and the output classification layer
         net_glob = MLP_tensor_simple(dim_in=args.dim_latent, dim_hidden=100, dim_out=args.num_classes).to(args.device)
         w_glob_keys = [net_glob.weight_keys[i] for i in [0]]
         w_glob_keys = list(itertools.chain.from_iterable(w_glob_keys))

    elif 'textcaps' in args.dataset and 'average' in args.model_type:
        # a simpler model with one global layer and the output classification layer
         net_glob = MLP_tensor_simple(dim_in=args.dim_latent, dim_hidden=100, dim_out=args.num_classes).to(args.device)
         w_glob_keys = [net_glob.weight_keys[i] for i in [0,1]]
         w_glob_keys = list(itertools.chain.from_iterable(w_glob_keys))

    elif 'bci' in args.dataset and 'classif' in args.model_type:
        net_glob = MLP_tensor_out(dim_in=args.dim_latent, dim_out=args.num_classes).to(args.device)
        w_glob_keys = []
        w_glob_keys = list(itertools.chain.from_iterable(w_glob_keys))
        
    elif 'bci' in args.dataset and 'no-hlayers' in args.model_type:
        # a simpler model with one global layer and the output classification layer
        net_glob = MLP_tensor_simple(dim_in=args.dim_latent, dim_hidden=100, dim_out=args.num_classes).to(args.device)
        w_glob_keys = [net_glob.weight_keys[i] for i in [0]]
        w_glob_keys = list(itertools.chain.from_iterable(w_glob_keys))
         
    else:
        exit('Error: unrecognized model')
    print(net_glob)

    return net_glob, w_glob_keys

def get_reg_model(args):
    if 'reg_toy_1' == args.dataset and 'linear' in args.model_type:
        net_glob = MLPReg_tensor_out(dim_in=args.dim_latent).to(args.device)
        # specify the representation parameters (in w_glob_keys) and head parameters (all others)
        w_glob_keys = []
        w_glob_keys = list(itertools.chain.from_iterable(w_glob_keys))

    return net_glob, w_glob_keys
 

def get_preproc_model(args, dim_in=2, dim_add=0,dim_out=2):
    #n_out = 2
  
    
    if args.dataset == 'toy_1' or args.dataset == 'toy_2' or args.dataset == 'toy_12' or args.dataset == 'toy_align':
        net_preproc = MLP_preproc(dim_in+dim_add,args.n_hidden, dim_out =  args.dim_latent)
    if args.dataset == 'toy_3' :
        net_preproc = MLP_preproc(dim_in,args.n_hidden, dim_out =  args.dim_latent)
    if args.dataset == 'textcaps' :
        net_preproc = MLP_preproc(dim_in,dim_hidden=args.n_hidden, dim_out =  args.dim_latent)
    if args.dataset == 'textcaps-clip' :
        net_preproc = MLP_preproc(dim_in,dim_hidden=args.n_hidden, dim_out =  args.dim_latent)
    elif args.dataset in ['bci-full','bci-subset-specific','bci-subset-common'] :
        net_preproc = MLP_preproc_drop(dim_in,dim_hidden=args.n_hidden, dim_out =  args.dim_latent)
    elif args.dataset == 'MU':
        if dim_add == 'usps':
            net_preproc = DigitsPreProc(dim_out=20,dim_latent=args.dim_latent)
        elif dim_add == 'mnist':
            net_preproc = DigitsPreProc(dim_out=320,dim_latent=args.dim_latent)
    elif args.dataset == 'MU-resize':
        net_preproc = DigitsPreProc(dim_out=320,dim_latent=args.dim_latent)
    elif args.dataset == 'femnist': 
        net_preproc = MLP_preproc(dim_in=784,dim_hidden=args.n_hidden, dim_out =  args.dim_latent)
    
    
    elif args.dataset == 'reg_toy_1':
        net_preproc = MLPReg_preproc(dim_in,args.n_hidden, dim_out =  args.dim_latent)

    return net_preproc




#-----------------------------------------------------------------------------
#       local function transformation for toy dataset and textcaps
#-----------------------------------------------------------------------------
class MLP_preproc(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP_preproc, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.layer_hidden1 = nn.Linear(dim_hidden, dim_hidden)
        self.layer_out = nn.Linear(dim_hidden,dim_out)
        self.weight_keys = [['layer_input.weight', 'layer_input.bias'],
                            ['layer_hidden1.weight', 'layer_hidden1.bias'],
                            ['layer_out.weight', 'layer_out.bias']
                            ]

    def forward(self, x):
        x = self.layer_input(x)
        x = self.relu(x)
        x = self.layer_hidden1(x)
        x = self.relu(x)
        x = self.layer_out(x)
        return x

class MLP_preproc_drop(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP_preproc_drop, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.layer_hidden1 = nn.Linear(dim_hidden, dim_hidden)
        self.layer_out = nn.Linear(dim_hidden,dim_out)
        self.weight_keys = [['layer_input.weight', 'layer_input.bias'],
                            ['layer_hidden1.weight', 'layer_hidden1.bias'],
                            ['layer_out.weight', 'layer_out.bias']
                            ]

    def forward(self, x):
        x = self.layer_input(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer_hidden1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        return x






#-----------------------------------------------------------------------------
#       local functions g_theta for toy dataset and textcaps
#-----------------------------------------------------------------------------


class MLP_tensor(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP_tensor, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.LeakyReLU()
        self.layer_hidden1 = nn.Linear(dim_hidden, dim_hidden)
        self.layer_out = nn.Linear(dim_hidden,dim_out)
        self.softmax = nn.Softmax(dim=1)
        self.weight_keys = [['layer_input.weight', 'layer_input.bias'],
                            ['layer_hidden1.weight', 'layer_hidden1.bias'],
                            #['layer_hidden2.weight', 'layer_hidden2.bias'],
                            ['layer_out.weight', 'layer_out.bias']
                            ]

    def forward(self, x):
        x = self.layer_input(x)
        x = self.relu(x)
        x = self.layer_hidden1(x)
        x = self.relu(x)
        x = self.layer_out(x)
        return x 

    
class MLP_tensor_simple(nn.Module):
    ##  the model n-hlayers 
    ##  layer_input is going to be shared across clients
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP_tensor_simple, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.LeakyReLU()
        self.layer_out = nn.Linear(dim_hidden,dim_out)
        self.weight_keys = [['layer_input.weight', 'layer_input.bias'],
                            ['layer_out.weight', 'layer_out.bias']
                            ]
   
    def forward(self, x):
        x = self.layer_input(x) 
        x = self.relu(x)
        x = self.layer_out(x)
        return x 

class MLP_tensor_out(nn.Module):
    ## the model classif
    ## everything is local in this model 
    def __init__(self, dim_in, dim_out):
        super(MLP_tensor_out, self).__init__()
        self.layer_out = nn.Linear(dim_in,dim_out)
        self.weight_keys = [['layer_out.weight', 'layer_out.bias']
                            ]

    def forward(self, x):

        x = self.layer_out(x)
        return x

    



#-----------------------------------------------------------------------------
#
#                   Model for Local and HetArch
#
#------------------------------------------------------------------------------


class MLP_HetNN(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, drop=False):
        super(MLP_HetNN, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.layer_hidden1 = nn.Linear(dim_hidden, dim_hidden)
        self.layer_out = nn.Linear(dim_hidden,dim_out)

        self.weight_keys = [['layer_input.weight', 'layer_input.bias'],
                            ['layer_hidden1.weight', 'layer_hidden1.bias'],
                            ['layer_out.weight', 'layer_out.bias']
                            ]

    def forward(self, x):
        #x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.relu(x)
        x = self.layer_hidden1(x)

        x_r = self.relu(x)  # useful for hetarch (alignment on pre-last layer)
        x = self.layer_out(x_r)
        return x, x_r

class MLP_HetNN_2(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP_HetNN_2, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.5)
        self.layer_hidden1 = nn.Linear(dim_hidden, dim_hidden)
        self.layer_hidden2 = nn.Linear(dim_hidden, dim_hidden)

        self.layer_out = nn.Linear(dim_hidden,dim_out)
        self.softmax = nn.Softmax(dim=1)
        self.weight_keys = [['layer_input.weight', 'layer_input.bias'],
                            ['layer_hidden1.weight', 'layer_hidden1.bias'],
                            ['layer_hidden2.weight', 'layer_hidden2.bias'],
                            ['layer_out.weight', 'layer_out.bias']
                            ]

    def forward(self, x):
        #x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer_hidden1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer_hidden2(x)
        x_r = self.relu(x)
        x = self.layer_out(x_r)
        return x, x_r
    



class DigitsPreProc(nn.Module):

    def __init__(self,dim_out,dim_latent):
        super(DigitsPreProc, self).__init__()        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()  
        self.fc1 = nn.Linear(dim_out, dim_latent)
        self.dim_out = dim_out
    def forward(self, input):
        x = torch.sigmoid(F.max_pool2d(self.conv1(input), 2)) 
        x = torch.sigmoid(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, self.dim_out)
        x = self.fc1(x)
        return x 

class DigitsClassifier(nn.Module):

    def __init__(self, dim_latent, num_classes):
        super(DigitsClassifier, self).__init__()
        self.layer_out = nn.Linear(dim_latent,num_classes)
        self.weight_keys = [['layer_out.weight', 'layer_out.bias']
                            ]
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        x = self.layer_out(x)
        return x


class DigitsHet(nn.Module):

    def __init__(self,dim_out,dim_latent,num_classes):
        super(DigitsHet, self).__init__()        

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()  #Dropout
        self.fc1 = nn.Linear(dim_out, dim_latent)
        self.layer_out = nn.Linear(dim_latent,num_classes)

        self.relu = nn.LeakyReLU()
        self.dim_out = dim_out
    def forward(self, input):
        x = torch.sigmoid(F.max_pool2d(self.conv1(input), 2)) 
        #Convolutional Layer/Dropout/Pooling Layer/Activation
        x = torch.sigmoid(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, self.dim_out)
        x = self.fc1(x)
        xr = self.relu(x)
        x = self.layer_out(xr)
        return x, xr

        return x 


#---------------------------------------------------------------------------
#
#               REGression
#
#---------------------------------------------------------------------------
class MLPReg_tensor_out(nn.Module):
    def __init__(self, dim_in,dim_out=1):
        super(MLPReg_tensor_out, self).__init__()
        self.layer_out = nn.Linear(dim_in,dim_out)
        self.weight_keys = [['layer_out.weight', 'layer_out.bias']
                            ]

    def forward(self, x):
 
        x = self.layer_out(x)
        return x 

class MLPReg_preproc(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLPReg_preproc, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.sigmoid = nn.LeakyReLU(0.2)
        self.layer_out = nn.Linear(dim_hidden,dim_out)


    def forward(self, x):
        x = self.layer_input(x)
        x = self.sigmoid(x)
        x = self.layer_out(x)
        x = self.sigmoid(x)

        return x

class MLPReg_HetNN(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, drop=False):
        super(MLPReg_HetNN, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.sigmoid = nn.LeakyReLU(0.2)

        self.layer_hidden1 = nn.Linear(dim_hidden, dim_hidden)
        #self.layer_hidden2 = nn.Linear(dim_hidden, dim_hidden)
        self.layer_out = nn.Linear(dim_hidden,1)
        self.weight_keys = [['layer_input.weight', 'layer_input.bias'],
                            ['layer_hidden1.weight', 'layer_hidden1.bias'],
                            #['layer_hidden2.weight', 'layer_hidden2.bias'],
                            ['layer_out.weight', 'layer_out.bias']
                            ]

    def forward(self, x):
        #x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.sigmoid(x)
        x = self.layer_hidden1(x)
        x_r = self.sigmoid(x)  # useful for hetarch (alignment on pre-last layer)
        x = self.layer_out(x_r)
        return x, x_r

