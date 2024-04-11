# Modified from: https://github.com/pliang279/LG-FedAvg/blob/master/main_fed.py
# credit goes to: Paul Pu Liang

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

# Adapted from the code of Collins et al.

# This code runs the competitors: 
#	* local learning
#	* adaptation of learning with heterogenous architecture Makhija
#
#
# Author a. rakotomamonjy

#%%

import random
import copy
import numpy as np
import torch
import os
from Het_Update import Het_LocalUpdate_Competitor
from Het_Update import  het_competitor_test_img_local_all
import argparse
import pandas as pd
import sys

import time
import warnings
warnings.filterwarnings("ignore")
if __name__ == '__main__':    

    if 0 :
        sys.argv = ['']
        print("no args")
    parser = argparse.ArgumentParser()

    
    parser.add_argument('--alg', type=str, default='hetarch', help="rounds of training")
    parser.add_argument('--dataset', type=str, default='femnist', help="rounds of training")


    parser.add_argument('--num_users', type=int, default=150, help="rounds of training")
    parser.add_argument('--shard_per_user', type=int, default=3, help="rounds of training")
    parser.add_argument('--num_classes', type=int, default=10, help="num of classes")

    parser.add_argument('--epochs', type=int, default=50, help="rounds of training")
    parser.add_argument('--lr', type=float, default=0.001, help="rounds of training")
    parser.add_argument('--local_bs', type=int, default=100, help="rounds of training")
    parser.add_argument('--local_ep', type=int, default=10, help="rounds of training")
    parser.add_argument('--frac', type=float, default=0.1,help="the fraction of clients: C")

    parser.add_argument('--mu', type=float, default=0.1, help="rounds of training")

    parser.add_argument('--n_hidden', type=int, default=64, help="rounds of training")
    parser.add_argument('--dim_latent', type=int, default=64, help="rounds of training")
    
    parser.add_argument('--seed', type=int, default=0, help="rounds of training")
    parser.add_argument('--gpu', type=int, default=-1, help="rounds of training")
    parser.add_argument('--num_workers', type=int, default=0, help="rounds of training")
    parser.add_argument('--test_freq', type=int, default=1, help="rounds of training")
    parser.add_argument('--timing', type=int, default=0, help="rounds of training")
    parser.add_argument('--Xrad_size', type=int, default=100, help="xrad size")
    parser.add_argument('--savedir', type=str, default='./save/', help="save dire")

    
    args = parser.parse_args()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True


    # fetching datasets for experiments
    if 'toy_1' == args.dataset:
        from Het_data import get_toy_1
        data_ = get_toy_1(args)
        user_data, user_data_addeddim = data_[0], data_[1]
        user_test_data,user_test_data_addeddim = data_[2], data_[3] 
    elif 'toy_2' == args.dataset:
        from Het_data import get_toy_2

        data_ = get_toy_2(args)
        user_data, user_data_addeddim = data_[0], data_[1]
        user_test_data,user_test_data_addeddim = data_[2], data_[3] 
    
    elif 'timing' == args.dataset:
        from Het_data import get_toy_2

        data_ = get_toy_2(args)
        user_data, user_data_addeddim = data_[0], data_[1]
        user_test_data,user_test_data_addeddim = data_[2], data_[3] 
        args.local_ep = 10    
        args.local_rep_ep = 5    ##### especially for large global shared model
        args.align_epochs = 0 # iteration of pretraining
        args.epochs = 10

    
    elif 'toy_3' == args.dataset:
        from Het_data import get_toy_3
        data_ = get_toy_3(args)
        user_data, user_data_addeddim = data_[0], data_[1]
        user_test_data,user_test_data_addeddim = data_[2], data_[3] 
    elif 'toy_12' == args.dataset:
        # added noisy features
        from Het_data import get_toy_12

        data_ = get_toy_12(args)
        user_data, user_data_addeddim = data_[0], data_[1]
        user_test_data,user_test_data_addeddim = data_[2], data_[3] 
 
    elif 'toy_align' == args.dataset:
        # added noisy features
        from Het_data import get_toy_align

        data_ = get_toy_align(args)
        user_data, user_data_addeddim = data_[0], data_[1]
        user_test_data,user_test_data_addeddim = data_[2], data_[3] 

    elif 'cifar10' == args.dataset:
        from Het_data import get_cifar10
        user_data, user_test_data = get_cifar10(args)
        args.dim = 64

    elif args.dataset == 'MU':
        from Het_data import get_mnist_usps
        args.num_classes = 10
        args.dim_latent = 64
        user_data, user_data_addeddim, user_test_data, user_data_addeddim = get_mnist_usps(args)
        args.local_ep = 10   
        args.local_rep_ep = 1  
        args.lr = 0.001
        args.local_bs = 100
    elif args.dataset == 'MU-resize':
        from Het_data import get_mnist_usps
        args.num_classes = 10
        args.dim_latent = 64
        args.image_size = 28
        user_data, user_data_addeddim, user_test_data, user_data_addeddim = get_mnist_usps(args)
        args.local_ep = 10   
        args.local_rep_ep = 1  
        args.lr = 0.001
        args.local_bs = 100

    elif args.dataset == 'textcaps':
        from Het_data import get_textcaps
        args.num_classes = 4
        user_data, user_data_addeddim, user_test_data, user_data_addeddim  = get_textcaps(args)

    elif args.dataset == 'textcaps-clip':
        from Het_data import get_textcaps_clip

        user_data, user_data_addeddim, user_test_data, user_data_addeddim  = get_textcaps_clip(args)
    elif args.dataset == 'bci-full':
        from Het_data import get_bci_full
        args.num_classes = 5
        user_data, user_data_addeddim, user_test_data, user_data_addeddim  = get_bci_full(args)
        args.num_users = len(user_data)
        args.shard_per_user = 2
        args.epochs = 50
        args.dim_latent = 64
        args.n_hidden = 64
        args.local_bs = 100
        args.lr = 0.001
        args.num_users = len(user_data)

        print(len(user_data))
        lens = np.ones(args.num_users)
    elif args.dataset == 'bci-subset-specific':
        from Het_data import get_bci_subset_specific
        args.num_classes = 5
        args.num_users = 40
        user_data, user_data_addeddim, user_test_data, user_data_addeddim  = get_bci_subset_specific(args)
        lens = np.ones(args.num_users)

    elif args.dataset == 'bci-subset-common':
        from Het_data import get_bci_subset_common
        args.num_classes = 5
        args.num_users = 40
        user_data, user_data_addeddim, user_test_data, user_data_addeddim  = get_bci_subset_common(args)
        lens = np.ones(args.num_users)

    elif args.dataset == 'isic2019':
        from Het_data import get_isic2019_enlarge
        args.num_classes = 8
        args.num_users = 14

        args.frac = 0.25
        args.epochs = 50

        user_data, user_data_addeddim, user_test_data, user_data_addeddim  = get_isic2019_enlarge(args)
    elif args.dataset == 'femnist':
        from Het_data import get_femnist
        
        print('hre',args.num_users)
        user_data, user_data_addeddim, user_test_data, user_data_addeddim  = get_femnist(args)
        print(args.num_users)
        args.num_classes = 10
        user_d = {}
        user_dt = {}
        for i in range(args.num_users):
            user_d[i] = user_data[i]
            user_dt[i] = user_test_data[i]
        user_data = user_d
        user_test_data = user_dt
        #args.frac=1
        #args.epochs = 500

        lens = np.ones(args.num_users)

    else:
        raise ValueError("unknown dataset")
    lens = np.ones(args.num_users)

    print(args.alg, args.dataset)
    print(args)


    #%%
    #
    # generate local models for each user and  adapt them for each target
    # local distributions

    from Het_Nets import MLP_HetNN, DigitsHet, MLP_HetNN_2
    nets_local = {}
    for user in range(args.num_users):
        if args.dataset == 'toy_1' or args.dataset == 'toy_2'  or args.dataset == 'toy_align'  or args.dataset == 'toy_12':
            nets_local[user] = MLP_HetNN(dim_in=args.dim+user_data_addeddim[user], dim_hidden= args.n_hidden, dim_out = args.num_classes)
        elif args.dataset == 'toy_3':
            nets_local[user] = MLP_HetNN(dim_in=user_data_addeddim[user], dim_hidden= args.n_hidden, dim_out = args.num_classes)
        elif args.dataset == 'textcaps' or args.dataset == 'textcaps-clip':
            nets_local[user] = MLP_HetNN(dim_in=user_data_addeddim[user], dim_hidden= args.n_hidden,
                                            dim_out = args.num_classes,
                                            drop = True)
        elif args.dataset in  ['bci-full','bci-subset-specific' ,'bci-subset-common']:
               nets_local[user] = MLP_HetNN_2(dim_in=user_data_addeddim[user], dim_hidden= args.n_hidden,dim_out = args.num_classes)
        elif args.dataset == 'MU':
            if user_data_addeddim[user] == 'usps':
                dim_out, dim_latent, num_classes = 20,100,10
            else:
                dim_out, dim_latent, num_classes = 320,100,10
            nets_local[user] =  DigitsHet(dim_out=dim_out,dim_latent=dim_latent,num_classes=num_classes)
        elif args.dataset == 'MU-resize':
            # all images are resized to 18x18
            dim_out, dim_latent, num_classes = 320,100,10
            nets_local[user] =  DigitsHet(dim_out=dim_out,dim_latent=dim_latent,num_classes=num_classes)
        
        elif args.dataset == 'isic2019':
            nets_local[user] = MLP_HetNN(dim_in=user_data_addeddim[user], dim_hidden= args.n_hidden, dim_out = args.num_classes)
        elif args.dataset in  ['femnist']:
            nets_local[user] = MLP_HetNN(dim_in=user_data_addeddim[user], dim_hidden= args.n_hidden,dim_out = args.num_classes)
    
    #%%
    loss_train = []
    accs = []
    accs_train = []
    times = []
    accs10 = 0
    accs10_glob = 0
    start = time.time()
    for iter in range(args.epochs+1):

        w_glob = {}
        loss_locals = []
        m = max(int(args.frac * args.num_users), 1)
        if iter == args.epochs:
            m = args.num_users

        #idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        times_in = []
        total_len=0
        
        #--------------------------------------------------------------------
        # compute RAD
        #
        if args.alg == 'hetarch':
            
            Xaux={}
            if 'toy_1' in args.dataset or 'toy_2' in args.dataset or 'toy_align' in args.dataset:
                Xrad_ = torch.randn(args.Xrad_size,args.dim).to(args.device)
            if 'toy_3' in args.dataset:
                Xrad_ = torch.randn(args.Xrad_size,args.dim_pertub_max).to(args.device)
        
            elif args.dataset=='MU':
                Xrad_ = torch.randn(args.Xrad_size,1,28,28).to(args.device)
            elif args.dataset=='MU-resize':
                Xrad_ = torch.randn(args.Xrad_size,1,28,28).to(args.device)
            elif args.dataset == 'textcaps':
                Xrad_ = torch.randn(args.Xrad_size,768).to(args.device)
            elif args.dataset == 'textcaps-clip':
                Xrad_ = torch.randn(args.Xrad_size,768).to(args.device)
            elif args.dataset == 'bci-full' or args.dataset == 'bci-subset-specific':
                Xrad_ = torch.randn(args.Xrad_size,1830).to(args.device)
            elif args.dataset == 'bci-subset-common':
                Xrad_ = torch.randn(args.Xrad_size,6).to(args.device)
            elif args.dataset == 'femnist':
                Xrad_ = torch.randn(args.Xrad_size,784).to(args.device)
            #TODO creer une lambda fonction pour Xrad en 
            K = torch.zeros(args.Xrad_size,args.Xrad_size).to(args.device)
            for idx in range(args.num_users):
                with torch.no_grad():
                    nets_local[idx].to(args.device)                
                    if 'toy_1' in args.dataset or 'toy_2' in args.dataset:
                        Xaux[idx] = torch.randn(args.Xrad_size,user_data_addeddim[idx]).to(args.device)
                        Xrad = torch.cat((Xrad_,Xaux[idx]),dim=1)
                    elif  'toy_3' in args.dataset:
                        Xrad = Xrad_[:,:user_data_addeddim[idx]]
                    elif args.dataset=='MU':
                        if user_data_addeddim[idx] == 'usps':
                            Xrad = Xrad_[:,:,0:16,0:16]
                        else:
                            Xrad = Xrad_
                    elif args.dataset=='femnist':
                            Xrad = Xrad_
                    elif args.dataset=='MU-resize':
                            Xrad = Xrad_
                    elif 'textcaps' in args.dataset:
                        # works for textcaps and textcaps-clip
                        Xrad =  Xrad_[:,:user_data_addeddim[idx]]
                    elif 'bci' in args.dataset:
                        # works for bci
                        Xrad =  Xrad_[:,:user_data_addeddim[idx]]
                    A_i =   nets_local[idx](Xrad)[1]
                    K += A_i @A_i.T     
            K /= args.num_users  #TODO change normalization
        elif 'local':
            K = None
            Xrad = None


        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for ind, idx in enumerate(idxs_users):
            #print(ind,'**')
            start_in = time.time()
            local = Het_LocalUpdate_Competitor(args=args, dataset= user_data[idx])

            net_local = nets_local[idx]
            #-----------------------------------------------------------------
            # preparing alignment data for HetArch local models
            #------------------------------------------------------------------
            if args.alg == 'hetarch' and args.dataset == 'toy_2':
                Xrad = torch.cat((Xrad_,Xaux[idx]),dim=1)
            
            if args.alg == 'hetarch' and args.dataset == 'toy_3':
                Xrad = Xrad_[:,:user_data_addeddim[idx]]
            elif args.alg == 'hetarch' and args.dataset == 'MU':
                if user_data_addeddim[idx] == 'usps':
                    Xrad = Xrad_[:,:,0:16,0:16]
                else:
                    Xrad = Xrad_
            elif args.alg == 'hetarch' and args.dataset == 'MU-resize':
                    Xrad = Xrad_
            elif args.alg == 'hetarch' and args.dataset == 'femnist':
                    Xrad = Xrad_
            elif args.alg == 'hetarch' and args.dataset == 'textcaps':
                Xrad = Xrad_[:,:user_data_addeddim[idx]]
            elif args.alg == 'hetarch' and 'bci' in args.dataset:
                Xrad = Xrad_[:,:user_data_addeddim[idx]]
            #-----------------------------------------------------------------
            # updating local models
            #------------------------------------------------------------------    
            last = iter == args.epochs
            #print(Xrad.shape,net_local)
            w_local, loss, indd = local.train(net=net_local.to(args.device), 
                                              K = K,
                                              Xrad = Xrad,
                                              lr=args.lr, last=last)
            loss_locals.append(copy.deepcopy(loss))
            total_len += lens[idx]
            
        
        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)


 
        times_in.append( time.time() - start_in )
        print('time:',times_in[-1])




      
        #
        #    verbosity
        #
   
        if iter % args.test_freq==args.test_freq-1 or iter>=args.epochs-10:
            if times == []:
                times.append(max(times_in))
            else:
                times.append(times[-1] + max(times_in))
            acc_test, loss_test,  bal_acc_test = het_competitor_test_img_local_all(nets_local, args,
                                                         user_test_data,
                                                         indd=indd)
            acc_train, loss_tr, bal_acc_train = het_competitor_test_img_local_all(nets_local, args,
                                                         user_data,
                                                         indd=indd)
            
            acc_train = bal_acc_train if args.dataset=='isic2019' else acc_train
            acc_test = bal_acc_test if args.dataset == 'isic2019' else acc_test
            
            accs.append(acc_test)
            accs_train.append(acc_train)

            # for algs which learn a single global model, these are the local accuracies (computed using the locally updated versions of the global model at the end of each round)
            if iter != args.epochs:
                print('Round {:3d}, Train loss: {:.3f}, Test loss: {:.3f}, Local Test accuracy: {:.2f}'.format(
                        iter, loss_avg, loss_test, acc_test))
            else:
                # in the final round, we sample all users, and for the algs which learn a single global model, we fine-tune the head for 10 local epochs for fair comparison with FedRep
                print('Final Round, Train loss: {:.3f}, Test loss: {:.3f}, Test accuracy: {:.2f}'.format(
                        loss_avg, loss_test, acc_test))
            if iter >= args.epochs-10 and iter != args.epochs:
                accs10 += acc_test/10


            if iter >= args.epochs-10 and iter != args.epochs:
                accs10_glob += acc_test/10

        # if iter % args.save_every==args.save_every-1:
        #     model_save_path = './save/accs_'+ args.alg + '_' + args.dataset + '_' + str(args.num_users) +'_'+ str(args.shard_per_user) +'_iter' + str(iter)+ '.pt'
        #     torch.save(net_glob.state_dict(), model_save_path)

    print('Average accuracy final 10 rounds: {}'.format(accs10))
    if args.alg == 'fedavg' or args.alg == 'prox':
        print('Average global accuracy final 10 rounds: {}'.format(accs10_glob))
    end = time.time()

    #%%    
    save_dir = args.savedir
    out_dir = f"{args.dataset}-{args.num_users:d}-{args.shard_per_user:d}-frac{args.frac:2.1f}/"
    opt = copy.deepcopy(vars(args))    
    for keys in ['full_client_freq','num_workers','device','mean_target_variance',
                  'align_epochs_altern','num_classes','subsample_client_data',
                  'gpu','test_freq','n_per_class','dataset','num_users','shard_per_user','frac','savedir']:
        opt.pop(keys, None)
    
    key_orig = ['local_ep','local_rep_ep','model_type','update_prior','update_net_preproc']
    key_new = ['l_ep','l_repep','mdl','upd_prior','upd_prprc']
    
    if not (os.path.isdir(save_dir+out_dir)):
        os.makedirs(save_dir+out_dir)
 
        
    
    filename = ""
    for key in opt.keys():
        val = str(opt[key])
        if key_orig.count(key)>0:
            filename += f"{key_new[key_orig.index(key)]}-{val}-"
        else:
            filename += f"{key}-{val}-" 

    base_dir = filename + '.csv'
    user_save_path = base_dir
    accs_m = np.array([accs,accs_train])
    times= np.array(times)

    accs_frame = pd.DataFrame({'accs_test': accs_m[0, :], 'accs_train': accs_m[1, :],'times':times})
    accs_frame.to_csv(save_dir+out_dir+base_dir, index=False)
    



    #%%
  
