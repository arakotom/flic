# Modified from: https://github.com/pliang279/LG-FedAvg/blob/master/models/Update.py
# credit: Paul Pu Liang

# For MAML (PerFedAvg) implementation, code was adapted from https://github.com/AntreasAntoniou/HowToTrainYourMAMLPytorch/blob/master/few_shot_learning_system.py
# credit: Antreas Antoniou

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import math
import torch.linalg as linalg

import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import copy
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score

#from models.language_utils import get_word_emb_arr, repackage_hidden, process_x, process_y 
#import ot


def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad

def dist_torch(x1,x2):
    x1p = x1.pow(2).sum(1).unsqueeze(1)
    x2p = x2.pow(2).sum(1).unsqueeze(1)
    prod_x1x2 = torch.mm(x1,x2.t())
    distance = x1p.expand_as(prod_x1x2) + x2p.t().expand_as(prod_x1x2) -2*prod_x1x2
    return distance #/x1.size(0)/x2.size(0) 


def calculate_mmd(X,Y):
    def my_cdist(x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)
    
    def gaussian_kernel(x, y, gamma=[0.0001,0.001, 0.01, 0.1, 1, 10, 100]):
        D = my_cdist(x, y)
        K = torch.zeros_like(D)
    
        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))
    
        return K
    

    Kxx = gaussian_kernel(X, X).mean()
    Kyy = gaussian_kernel(Y, Y).mean()
    Kxy = gaussian_kernel(X, Y).mean()
    return Kxx + Kyy - 2 * Kxy


def calculate_2_wasserstein_dist(X, Y):
    '''
    Calulates the two components of the 2-Wasserstein metric:
    The general formula is given by: d(P_X, P_Y) = min_{X, Y} E[|X-Y|^2]
    For multivariate gaussian distributed inputs z_X ~ MN(mu_X, cov_X) and z_Y ~ MN(mu_Y, cov_Y),
    this reduces to: d = |mu_X - mu_Y|^2 - Tr(cov_X + cov_Y - 2(cov_X * cov_Y)^(1/2))
    Fast method implemented according to following paper: https://arxiv.org/pdf/2009.14075.pdf
    Input shape: [b, n] (e.g. batch_size x num_features)
    Output shape: scalar
    '''


    if X.shape != Y.shape:
        raise ValueError("Expecting equal shapes for X and Y!")

    # the linear algebra ops will need some extra precision -> convert to double
    X, Y = X.transpose(0, 1).double(), Y.transpose(0, 1).double()  # [n, b]
    mu_X, mu_Y = torch.mean(X, dim=1, keepdim=True), torch.mean(Y, dim=1, keepdim=True)  # [n, 1]
    n, b = X.shape
    fact = 1.0 if b < 2 else 1.0 / (b - 1)

    # Cov. Matrix
    E_X = X - mu_X
    E_Y = Y - mu_Y
    cov_X = torch.matmul(E_X, E_X.t()) * fact  # [n, n]
    cov_Y = torch.matmul(E_Y, E_Y.t()) * fact

    # calculate Tr((cov_X * cov_Y)^(1/2)). with the method proposed in https://arxiv.org/pdf/2009.14075.pdf
    # The eigenvalues for M are real-valued.
    C_X = E_X * math.sqrt(fact)  # [n, n], "root" of covariance
    C_Y = E_Y * math.sqrt(fact)
    M_l = torch.matmul(C_X.t(), C_Y)
    M_r = torch.matmul(C_Y.t(), C_X)
    M = torch.matmul(M_l, M_r)
    
    S = linalg.eigvals(M+1e-6) + 1e-15  # add small constant to avoid infinite gradients from sqrt(0)
    sq_tr_cov = S.sqrt().abs().sum()

    # plug the sqrt_trace_component into Tr(cov_X + cov_Y - 2(cov_X * cov_Y)^(1/2))
    trace_term = torch.trace(cov_X + cov_Y) - 2.0 * sq_tr_cov  # scalar

    # |mu_X - mu_Y|^2
    diff = mu_X - mu_Y  # [n, 1]
    mean_term = torch.sum(torch.mul(diff, diff))  # scalar

    # put it together
    return (trace_term + mean_term).float()




def wass_loss(net_projector,data, label, prior,optimize_projector=True,distance='wd'):
    loss = 0
    present_label = torch.unique(label)
    for curr_label in present_label:                          
        ind_l = torch.where(label==curr_label)[0]
        #print(ind_l.shape,label)
        if ind_l.shape[0]>0:
            if optimize_projector:
                #print(data[ind_l].shape)
                out = net_projector(data[ind_l])
    
                with torch.no_grad():
                    mean_t = prior.mu[curr_label].detach()
                    var_t = prior.logvar[curr_label].detach()
                        
                prior_samples  = prior.sampling_gaussian(out.shape[0], mean_t, var_t) 
                if distance == 'wd':
                    loss +=  calculate_2_wasserstein_dist(out,prior_samples)
                elif distance == 'mmd':       
                    loss += calculate_mmd(out, prior_samples)
    
            else:
                # we optimize the local means of the priors, so we work on local
                # vectors
                set_requires_grad(net_projector,requires_grad=False)
                with torch.no_grad():
                    out = net_projector(data[ind_l])
    
    
                prior_samples  = prior.sampling_gaussian(out.shape[0], prior.mu_local[curr_label], prior.logvar[curr_label]) 
    
                if distance == 'wd':
                    loss +=  calculate_2_wasserstein_dist(out,prior_samples)
                elif distance == 'mmd':
                    loss += calculate_mmd(out, prior_samples)
    return loss






def train_preproc(net_preprocs,user_data,prior,n_epochs,args=None,verbose=True):
    
    prior.mu = prior.mu.to(args.device)  
    prior.logvar = prior.logvar.to(args.device)  
    if args.device == 'cuda':
        pin_memory = True
    else:
        pin_memory = False


    # for each user, optimize the loss
    idxs_users = np.arange(args.num_users)

    for ind, idx in enumerate(idxs_users):
        data_adapt = DataLoader(dataset=user_data[idx], batch_size=args.align_bs, shuffle=True
                                    ,drop_last=False,pin_memory=pin_memory,
                                    num_workers=args.num_workers)
        net_projector =  copy.deepcopy(net_preprocs[idx].to(args.device))
        set_requires_grad(net_projector, requires_grad=True)
        
        optimizer_preproc = torch.optim.Adam(net_projector.parameters(),lr=args.align_lr)

        for itm in range(n_epochs):

    
            loss_tot = 0
            for it, (data,label) in enumerate((data_adapt)):
                data, label = data.to(args.device), label.to(args.device)                        
                optimizer_preproc.zero_grad()
                net_projector.zero_grad()
                loss = wass_loss(net_projector,data, label, prior,optimize_projector=True,distance=args.distance)
                loss.backward()
                optimizer_preproc.step()

                loss_tot +=loss.item()
            if verbose :
                print(ind,itm,loss_tot)
        set_requires_grad(net_projector, requires_grad=False)
        net_preprocs[idx] = copy.deepcopy(net_projector)



class Het_LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, indd=None,
                 mean_target=None,current_iter= 1000):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()

        # generate a dataloader from tensor data            
        self.ldr_train = DataLoader(dataset, batch_size=self.args.local_bs, shuffle=True,
                                    pin_memory=True,
                                    num_workers=args.num_workers)
         
      
        
        self.dataset = dataset
        self.idxs = idxs
        self.indd = indd
        self.update_prior = args.update_prior > 0
        self.update_net_preproc = args.update_net_preproc > 0
        self.update_global_representation = current_iter > args.start_optimize_rep
    def train(self, net,net_preproc, w_glob_keys, last=False, dataset_test=None, 
              prior=None,ind=-1, idx=-1, lr=0.1):
        bias_p=[]
        weight_p=[]
        for name, p in net.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        optimizer = torch.optim.Adam(
        [     
            {'params': weight_p, 'weight_decay':lr},
            {'params': bias_p, 'weight_decay':0}
        ],
        lr=lr
        )
        # number of local epoch        
        local_eps = self.args.local_ep
        if last:
            local_eps =  max(10,local_eps-self.args.local_rep_ep)
        if self.update_global_representation:
            # part of the local eps are dedicated to the optimization of the
            # head (head_eps)
            head_eps = local_eps-self.args.local_rep_ep
        else:
            # we do not update the global representation hence,
            # all local epochs are dedicated to the head
            head_eps = local_eps
        epoch_loss = []
        num_updates = 0
        
        # optimizer for mu_local
        mu_local = nn.Parameter(prior.mu.clone(), requires_grad=True)
        optim_mean = torch.optim.Adam([mu_local],lr=0.001)
        
        
        net.to(self.args.device)
        net_preproc.to(self.args.device)
        mu_local = mu_local.to(self.args.device)
        prior.mu_temp = prior.mu_temp.to(self.args.device)  
        optimizer_preproc = torch.optim.Adam(net_preproc.parameters(),lr=self.args.align_lr)


        for iter in range(local_eps):

            net_preproc.train()
            net.train()
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                
                # iterations on local epoch
                if (iter < head_eps ) or last or not w_glob_keys:
                    if self.update_net_preproc:
                        set_requires_grad(net_preproc, requires_grad=True)
    
                    for name, param in net.named_parameters():
                        #print(name)
                        if name in w_glob_keys:
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
 
    
                elif (iter >= head_eps ):
                    # update only the global model
                    set_requires_grad(net_preproc, requires_grad=False)
    
                    for name, param in net.named_parameters():
                        if name in w_glob_keys:
                            param.requires_grad = True
                        else:
                            param.requires_grad = False
                    


                images, labels = images.to(self.args.device), labels.to(self.args.device)
                
                
                im_out = net_preproc(images) 
                log_probs = net(im_out)
                # classification loss
                loss_class = self.loss_func(log_probs, labels)

                # # alignement loss
                if self.update_net_preproc:
                    loss_W = wass_loss(net_preproc,images, labels, prior,optimize_projector=True)
                else:
                    loss_W = 0
                # computing classification loss on prior data
                loss_ref_dist = 0
                present_label = torch.unique(labels)
                for curr_label in present_label:                      
                    prior_samples  = prior.sampling_gaussian(images.shape[0], prior.mu[curr_label], prior.logvar[curr_label]) 
                    log_probs_prior_samples = net(prior_samples)
                    loss_ref_dist += self.loss_func(log_probs_prior_samples, curr_label*torch.ones(images.shape[0]).long().
                                                    to(self.args.device)) 


                num_updates += 1
                loss =  loss_class + self.args.reg_w*loss_W + self.args.reg_class_prior*loss_ref_dist
                batch_loss.append(loss.item())
                optimizer.zero_grad()
                optimizer_preproc.zero_grad()
                loss.backward()
                optimizer.step()
                if self.update_net_preproc:
                    optimizer_preproc.step()

                # #----------------------------------------------------------------
                # #    update mean of prior wrt to the classifier and the wass loss
                # #------------------------------------------------------------------
                if self.update_prior:
                    # mu_local are trainable parameter while prior.mu is not
                    # loss is computed on mu_local
                    prior.mu_local = mu_local
                    lossW = wass_loss(net_preproc,images, labels, prior,optimize_projector=False)
                    
                    set_requires_grad(net, requires_grad=False)
                    log_probs = net(mu_local)  
                    labels = torch.Tensor([ii for ii in range(self.args.num_classes)]).long()
                    labels = labels.to(self.args.device)
                    loss = self.loss_func(log_probs, labels) + lossW
                    optim_mean.zero_grad()
                    loss.backward()
                    optim_mean.step()
                    # restoring gradient of global model
                    for name, param in net.named_parameters():
                        if name in w_glob_keys:
                            param.requires_grad = False
                        else:
                            param.requires_grad = True


            
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            #print('**',iter,epoch_loss[-1])
        set_requires_grad(net_preproc, requires_grad=True)
        #% add to mu_temp for future averaging 
        prior.mu_temp += mu_local.detach()
        prior.n_update += 1
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), self.indd,epoch_loss
    def train_old(self, net,net_preproc, w_glob_keys, last=False, dataset_test=None, 
              prior=None,ind=-1, idx=-1, lr=0.1):
        bias_p=[]
        weight_p=[]
        for name, p in net.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        optimizer = torch.optim.Adam(
        [     
            {'params': weight_p, 'weight_decay':lr},
            {'params': bias_p, 'weight_decay':0}
        ],
        lr=lr
        )

        # number of local epoch        
        local_eps = self.args.local_ep
        if last:
            local_eps =  max(10,local_eps-self.args.local_rep_ep)
        if self.update_global_representation:
            # part of the local eps are dedicated to the optimization of the
            # head (head_eps)
            head_eps = local_eps-self.args.local_rep_ep
        else:
            # we do not update the global representation hence,
            # all local epochs are dedicated to the head
            head_eps = local_eps
        epoch_loss = []
        num_updates = 0
        
        mu_local = nn.Parameter(prior.mu.clone(), requires_grad=True)
        optim_mean = torch.optim.Adam([mu_local],lr=0.001)
        net.to(self.args.device)
        net_preproc.to(self.args.device)
        mu_local = mu_local.to(self.args.device)
        prior.mu_temp = prior.mu_temp.to(self.args.device)  


        for iter in range(local_eps):
            #-----------------------------------------------------------------
            #       OLD
            #-----------------------------------------------------------------

            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                
                # iterations on local epoch
                if (iter < head_eps ) or last or not w_glob_keys:
                    if self.update_net_preproc:
                        set_requires_grad(net_preproc, requires_grad=True)
    
                    for name, param in net.named_parameters():
                        #print(name)
                        if name in w_glob_keys:
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
 
    
                elif (iter > head_eps ):
                    # update only the global model
                    set_requires_grad(net_preproc, requires_grad=False)
    
                    for name, param in net.named_parameters():
                        if name in w_glob_keys:
                            param.requires_grad = True
                        else:
                            param.requires_grad = False
                
                #-----------------------------------------------------------------
                #       OLD
                #-----------------------------------------------------------------



                images, labels = images.to(self.args.device), labels.to(self.args.device)
                #with torch.no_grad():
                im_out = net_preproc(images) 
                net.zero_grad()
                log_probs = net(im_out)
                loss = self.loss_func(log_probs, labels)

                num_updates += 1
                batch_loss.append(loss.item())

                

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            #-----------------------------------------------------------------
            #       OLD
            #-----------------------------------------------------------------

                # #----------------------------------------------------------------
                # #    update mean of prior wrt to the classifier and the wass loss
                # #------------------------------------------------------------------
                if self.update_prior:
                    prior.mu_local = mu_local
                    lossW = wass_loss(net_preproc,images, labels, prior,optimize_projector=False)
                    #lossW = torch.zeros(1)
                    
                    set_requires_grad(net, requires_grad=False)
                    log_probs = net(mu_local)  
                    labels = torch.Tensor([ii for ii in range(self.args.num_classes)]).long()
                    labels = labels.to(self.args.device)
                    loss = self.loss_func(log_probs, labels) + lossW
                    optim_mean.zero_grad()
                    loss.backward()
                    optim_mean.step()
                    # restoring gradient of global model
                    for name, param in net.named_parameters():
                        if name in w_glob_keys:
                            param.requires_grad = False
                        else:
                            param.requires_grad = True


            
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            #print('**',iter,epoch_loss[-1])
        set_requires_grad(net_preproc, requires_grad=True)
        #% add to mu_temp for future averaging 
        prior.mu_temp += mu_local.detach()
        prior.n_update += 1
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), self.indd,epoch_loss



def het_test_img_local(net_g, net_preproc, user_data, args,idx=None,indd=None, user_idx=-1, idxs=None):
    net_g.eval()
    net_preproc.eval()
    test_loss = 0
    correct = 0
    net_preproc.to(args.device)
    net_g.to(args.device)


    # if leaf:
    #     data_loader = DataLoader(DatasetSplit_leaf(datatest_new,np.ones(len(datatest_new))), batch_size=args.local_bs, shuffle=False)
    # else:
    #     data_loader = DataLoader(DatasetSplit(dataset,idxs), batch_size=args.local_bs,shuffle=False)
    # if 'sent140' in args.dataset:
    #     hidden_train = net_g.init_hidden(args.local_bs)
    count = 0
    data_loader = DataLoader(user_data, batch_size=200, shuffle=True,drop_last=False)
    for idx, (data, target) in enumerate(data_loader):

        if args.gpu != -1:
            data, target = data.to(args.device), target.to(args.device)
        log_probs = net_g(net_preproc(data))
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
        count += data.shape[0]

        if idx==0:
            target_all = target.detach().cpu()
            y_pred_all = y_pred.detach().cpu()
        else:
            target_all = torch.cat((target_all,target.detach().cpu()),dim=0)
            y_pred_all = torch.cat((y_pred_all,y_pred.detach().cpu()),dim=0)
                
    test_loss /= count
    accuracy = 100.00 * float(correct) / count
    bal_acc = 100*balanced_accuracy_score(target_all, y_pred_all.long())
    net_g.train()
    net_preproc.train()
    
    return  accuracy, test_loss, bal_acc

def het_test_img_local_all(net, net_preprocs, args, users_test_data,w_locals=None
                           ,w_glob_keys=None, indd=None,dataset_train=None,dict_users_train=None, return_all=False):
    tot = 0
    num_idxxs = args.num_users
    acc_test_local = np.zeros(num_idxxs)
    loss_test_local = np.zeros(num_idxxs)
    bal_acc_test_local = np.zeros(num_idxxs)

    for idx in range(num_idxxs):
        # net is the global network.
        # it is copied and then local layers are overwritten.
        #print(idx)
        net_local = copy.deepcopy(net)
        if w_locals is not None:
            w_local = net_local.state_dict()
            for k in w_locals[idx].keys():
                if w_glob_keys is not None and k not in w_glob_keys:
                    w_local[k] = w_locals[idx][k]
                elif w_glob_keys is None:
                    w_local[k] = w_locals[idx][k]
            net_local.load_state_dict(w_local)
        net_local.eval()
        net_preproc = net_preprocs[idx]
        net_preproc.eval()

        a, b,c  = het_test_img_local(net_local,net_preproc, users_test_data[idx], args, user_idx=idx) 
        if 'toy' in args.dataset or 'textcaps' in args.dataset or  'bci' in args.dataset or 'femnist' in args.dataset:
            n_test = users_test_data[idx].tensors[0].shape[0]
        else:
            n_test = len(users_test_data[idx].idxs)
        tot += n_test
        acc_test_local[idx] = a*n_test
        loss_test_local[idx] = b*n_test
        bal_acc_test_local[idx] = c*n_test

        del net_local
        net_preproc.train()
    if return_all:
        return acc_test_local, loss_test_local
    return  sum(acc_test_local)/tot, sum(loss_test_local)/tot,  sum(bal_acc_test_local)/tot



#-----------------------------------------------------------------------------
#
#               update for local models and 
#
#-----------------------------------------------------------------------------


class Het_LocalUpdate_Competitor(object):
    def __init__(self, args, dataset=None, idxs=None, indd=None,
                 mean_target=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()

        self.ldr_train = DataLoader(dataset, batch_size=self.args.local_bs, shuffle=True)
         
      
        
        self.dataset = dataset
        self.idxs = idxs
        self.indd = indd
        # if mean_target == None:
        #     self.mean_target = torch.randn(args.num_classes,args.dim)*3

    def train(self, net,last=False, K=None, Xrad=None, dataset_test=None, 
              ind=-1, idx=-1, lr=0.1):
        bias_p=[]
        weight_p=[]
        for name, p in net.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        # optimizer = torch.optim.SGD(
        # [     
        #     {'params': weight_p, 'weight_decay':0.0001},
        #     {'params': bias_p, 'weight_decay':0}
        # ],
        # lr=lr, momentum=0.5
        # )
        optimizer = torch.optim.Adam(
        [     
            {'params': weight_p, 'weight_decay':0.0001},
            {'params': bias_p, 'weight_decay':0}
        ],
        lr=lr
        )

        local_eps = self.args.local_ep

        
        #head_eps = local_eps-self.args.local_rep_ep
        epoch_loss = []
        num_updates = 0
        net.to(self.args.device)
        for iter in range(local_eps):
            done = False
            net.train()
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):

                images, labels = images.to(self.args.device), labels.to(self.args.device)

                net.zero_grad()
                log_probs, _ = net(images)
                if Xrad is not None:
                    _, features = net(Xrad)

                    K_local = features@features.T
                    normK = torch.norm(K_local@K_local,p="fro")
                    if normK < 1:
                        normK = 1
                    reg = torch.norm(K@K_local,p="fro")**2/torch.norm(K@K.T,p="fro")/normK
                else:
                    reg = torch.zeros(1)
                #print(reg)
                loss = self.loss_func(log_probs, labels) + self.args.mu* reg.to(self.args.device)
                loss.backward()
                optimizer.step()
                num_updates += 1
                batch_loss.append(loss.item())
                # if num_updates == self.args.local_updates:
                #     done = True
                #     break
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            if done:
                break
            
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        #print(net.state_dict())
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), self.indd

def het_competitor_test_img_local_all(nets_local, args, users_test_data,
                                  indd=None,dataset_train=None,dict_users_train=None, return_all=False):
    tot = 0
    num_idxxs = args.num_users
    acc_test_local = np.zeros(num_idxxs)
    loss_test_local = np.zeros(num_idxxs)
    bal_acc_test_local = np.zeros(num_idxxs)

    for idx in range(num_idxxs):
        # net is the global network.
        # it is copied and then local layers are overwritten.
        #print(idx)
        net_local = nets_local[idx]
        net_local.eval()

        a, b,c = het_competitor_test_img_local(net_local, users_test_data[idx], args, user_idx=idx) 
        if 'toy' in args.dataset or 'textcaps' in args.dataset or  'bci' in args.dataset or 'femnist' in args.dataset:
            n_test = users_test_data[idx].tensors[0].shape[0]
        else:
            n_test = len(users_test_data[idx].idxs)
        tot += n_test

        acc_test_local[idx] = a*n_test
        loss_test_local[idx] = b*n_test
        bal_acc_test_local[idx] = c*n_test

        net_local.train()
    if return_all:
        return acc_test_local, loss_test_local
    return  sum(acc_test_local)/tot, sum(loss_test_local)/tot, sum(bal_acc_test_local)/tot


def het_competitor_test_img_local(net_g, user_data, args,idx=None,indd=None, user_idx=-1, idxs=None):
    net_g.eval()
    net_g.to(args.device)
    test_loss = 0
    correct = 0



    #     hidden_train = net_g.init_hidden(args.local_bs)
    count = 0
    data_loader = DataLoader(user_data, batch_size=200, shuffle=True,drop_last=False)
    for idx, (data, target) in enumerate(data_loader):

        if args.gpu != -1:
            data, target = data.to(args.device), target.to(args.device)
        log_probs = net_g((data))[0]
        # sum up batch loss

            
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
        count += data.shape[0]

        if idx==0:
            target_all = target.detach().cpu()
            y_pred_all = y_pred.detach().cpu()
        else:
            target_all = torch.cat((target_all,target.detach().cpu()),dim=0)
            y_pred_all = torch.cat((y_pred_all,y_pred.detach().cpu()),dim=0)
            
    test_loss /= count
    accuracy = 100.00 * float(correct) / count
    bal_acc = 100*balanced_accuracy_score(target_all, y_pred_all.long())

    net_g.train()
    return  accuracy, test_loss, bal_acc
