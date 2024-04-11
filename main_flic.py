# Modified from: https://github.com/pliang279/LG-FedAvg/blob/master/main_fed.py
# credit goes to: Paul Pu Liang

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

# Adapted from the code of Collins et al.
#
# Author A.R
#%%
import argparse
import os
import random
import copy
import numpy as np
import torch
import pandas as pd

from Het_Update import Het_LocalUpdate, het_test_img_local_all, train_preproc
from Het_Nets import  get_model, get_preproc_model


import time
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    import sys
    #sys.argv = ['']
    parser = argparse.ArgumentParser()

    parser.add_argument('--alg', type=str, default='FLic', help="Algorithm")
    parser.add_argument('--dataset', type=str, default='bci-subset-specific', help="choice of the dataset")

    parser.add_argument('--num_users', type=int, default=50, help="number of users")
    parser.add_argument('--shard_per_user', type=int, default=2, help="number of classes per user")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")

    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")

    parser.add_argument('--epochs', type=int, default=50,help="rounds of training")
    parser.add_argument('--local_bs', type=int, default=100, help="local batch size")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate for model")
    parser.add_argument('--local_ep', type=int, default=10, help="number of local epoch")
    parser.add_argument('--local_rep_ep', type=int, default=1, help="number of local epoch for representation among local_ep")
    parser.add_argument('--reg_w', type=float, default=0.001, help="regularization of W ")
    parser.add_argument('--reg_class_prior', type=float, default=0.001, help="regularization of W ")


    parser.add_argument('--model_type', type=str, default='classif', help="choosing the global model, [classif, no-hlayers, 2-hlayers]")
    parser.add_argument('--n_hidden', type=int, default=64, help="number of units in hidden layers")
    parser.add_argument('--dim_latent', type=int, default=64, help="latent dimension")
    parser.add_argument('--align_epochs', type=int, default=100, help="number of epochs for alignment during pretraining")
    parser.add_argument('--align_epochs_altern', type=int, default=1, help="number of epochs for alignment during alternate")
    parser.add_argument('--align_lr', type=float, default=0.001, help="learning rate of alignment ")
    parser.add_argument('--align_bs', type=int, default=10, help="batch_size for alignment")
    parser.add_argument('--distance', type=str, default='wd', help="distance for alignment")

    parser.add_argument('--mean_target_variance', type=int, default=10, help="std of random prior means")
    parser.add_argument('--update_prior', type=int, default=1, help= "updating prior (1 for True)")
    parser.add_argument('--update_net_preproc', type=int, default=1, help= "updating preproc network")
    parser.add_argument('--start_optimize_rep', type=int, default=20, help= "starting iterations for global model optim")

    parser.add_argument('--seed', type=int, default=0,help="choice of seed")
    parser.add_argument('--gpu', type=int, default=-1, help="gpu to use (if any")
    parser.add_argument('--num_workers', type=int, default=0, help="number of workers in dataloader")
    parser.add_argument('--test_freq', type=int, default=1, help="frequency of test eval")
    parser.add_argument('--timing', type=int, default=0, help="compute running time")
    parser.add_argument('--savedir', type=str, default='./save/', help="save dire")


    args = parser.parse_args()


    
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    
    lens = np.ones(args.num_users)
    if 'toy_2' == args.dataset:
        # added noisy features
        from Het_data import get_toy_2
        args.shard_per_user = 3
        data_ = get_toy_2(args)
        user_data, user_data_addeddim = data_[0], data_[1]
        user_test_data,user_test_data_addeddim = data_[2], data_[3] 

    elif 'toy_3' == args.dataset:
        # linear transformation
        from Het_data import get_toy_3
        data_ = get_toy_3(args)
        user_data, user_data_addeddim = data_[0], data_[1]
        user_test_data,user_test_data_addeddim = data_[2], data_[3] 
 
    
    elif args.dataset == 'MU':
        from Het_data import get_mnist_usps
        args.num_classes = 10
        user_data, user_data_addeddim, user_test_data, user_data_addeddim = get_mnist_usps(args)
        args.local_ep = 10    
        args.local_rep_ep = 1
        args.lr = 0.001
        args.local_bs = 100
        args.align_bs = 100
    elif args.dataset == 'MU-resize':
        from Het_data import get_mnist_usps
        args.num_classes = 10
        args.image_size = 28
        user_data, user_data_addeddim, user_test_data, user_data_addeddim = get_mnist_usps(args)
        args.local_ep = 10    
        args.local_rep_ep = 1
        args.lr = 0.001
        args.local_bs = 100
        args.align_bs = 100


    elif args.dataset == 'textcaps':
        from Het_data import get_textcaps
        args.num_classes = 4
        user_data, user_data_addeddim, user_test_data, user_data_addeddim  = get_textcaps(args)
        args.align_bs = 10

    elif args.dataset == 'textcaps-clip':
        from Het_data import get_textcaps_clip

        user_data, user_data_addeddim, user_test_data, user_data_addeddim  = get_textcaps_clip(args)

    elif args.dataset == 'bci-full':
        from Het_data import get_bci_full
        args.num_classes = 5

        user_data, user_data_addeddim, user_test_data, user_data_addeddim  = get_bci_full(args)
        args.num_users = len(user_data)

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
    elif args.dataset == 'femnist':
        from Het_data import get_femnist
        user_data, user_data_addeddim, user_test_data, user_data_addeddim  = get_femnist(args)
        args.num_classes = 10

        user_d = {}
        user_dt = {}
        for i in range(args.num_users):
            user_d[i] = user_data[i]
            user_dt[i] = user_test_data[i]
        user_data = user_d
        user_test_data = user_dt
            
        lens = np.ones(args.num_users)

    else:
        raise ValueError("unknown dataset")
    
    print(args)
    if args.model_type =='average':
        # for FedAverage, use one local epoch and use it for the global part
        # and the appropriate model
        args.local_ep = 1
        args.local_rep_ep = 1    

    
    
    
    # - select global model and trigger the train mode
    #   set appropriate arguments to make it trainable
    # - show model
    
    net_glob, w_glob_keys = get_model(args)
    net_glob.train()
    
    # if args.model_type != 'classif':
    #     net_glob.layer_input.weight = torch.nn.Parameter(torch.eye(args.dim_latent))
    #     net_glob.layer_input.requires_grad_(False)
    
    
    total_num_layers = len(net_glob.state_dict().keys())
    print(net_glob.state_dict().keys())
    
    
    # generate list of local models for each user
    #
    # copy the global model to all user
    
    w_locals = {}
    for user in range(args.num_users):
        w_local_dict = {}
        for key in net_glob.state_dict().keys():
            w_local_dict[key] = net_glob.state_dict()[key]
        w_locals[user] = w_local_dict
        
    # init reference distributions
        
    from Het_Prior import Prior
    prior = Prior([args.num_classes,args.dim_latent],args.mean_target_variance)
    
    #
    # generate local projectors  models for each user and pretrain them with
    # respect to target distributions
    #
    
    net_preprocs = {}
    for user in range(args.num_users):
        if args.dataset == 'toy_1' or args.dataset == 'toy_2' or args.dataset == 'toy_align':
            net_preproc = get_preproc_model(args, dim_in=args.dim, dim_add = user_data_addeddim[user], dim_out = args.dim_latent)
        elif args.dataset == 'toy_3':
            net_preproc = get_preproc_model(args, dim_in=user_data_addeddim[user], dim_out = args.dim_latent)
    
        elif args.dataset == 'MU':
            net_preproc = get_preproc_model(args, dim_add = user_data_addeddim[user], dim_out = args.dim_latent)
        elif args.dataset == 'MU-resize':
            net_preproc = get_preproc_model(args, dim_add = user_data_addeddim[user], dim_out = args.dim_latent)
        elif args.dataset == 'femnist':
            net_preproc = get_preproc_model(args, dim_add = user_data_addeddim[user], dim_out = args.dim_latent)
        elif args.dataset == 'textcaps':
            net_preproc = get_preproc_model(args, dim_in = user_data_addeddim[user], dim_out = args.dim_latent)
        elif args.dataset == 'textcaps-clip':
            net_preproc = get_preproc_model(args, dim_in = user_data_addeddim[user], dim_out = args.dim_latent)
        elif args.dataset in ['bci-full','bci-subset-specific' ,'bci-subset-common']:
            net_preproc = get_preproc_model(args, dim_in = user_data_addeddim[user], dim_out = args.dim_latent)
        
        else:
            net_preproc = get_preproc_model(args)
            args.dim,args.num_classes = 64,10
        net_preprocs[user]=(net_preproc)
    
    #%%
    print(net_preprocs[0])

    train_preproc(net_preprocs,user_data,prior,
                  n_epochs=args.align_epochs,
                  args=args,
                  verbose=True)

    #%%
    
    
    indd = None   
    loss_train = []
    loss_local_full = [[] for _ in range(args.num_users)]
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
            

        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        w_keys_epoch = w_glob_keys
        times_in = []
        total_len=0
        
        
        for ind, idx in enumerate(idxs_users):

            print(ind,'user:',idx)
            start_in = time.time()
            
            # instantiate an algo for localupdate
            local = Het_LocalUpdate(args=args, dataset= user_data[idx],current_iter=iter)
            local.update_prior = args.update_prior 
            #-----------------------------------------------------------------
            # starting local update
            # 1. initialize a curent local model (the one of the user to be update) with the global model 
            # 2. replace weights that are local with current status of local model for that users 
            # 3. update local model net_local -- 
            #-----------------------------------------------------------------
            net_local = copy.deepcopy(net_glob)
            w_local = net_local.state_dict()
            for k in w_locals[idx].keys():
                if k not in w_glob_keys:
                    w_local[k] = w_locals[idx][k]
            net_local.load_state_dict(w_local)
            #-----------------------------------------------------------------
            # updating local models
            #------------------------------------------------------------------
            

            last = iter == args.epochs
            w_local, loss, indd, last_loss = local.train(net=net_local.to(args.device), 
                                              net_preproc=net_preprocs[idx].to(args.device),
                                              w_glob_keys=w_glob_keys, 
                                              prior=prior,
                                              lr=args.lr, last=last)
            loss_locals.append(copy.deepcopy(loss))
            total_len += lens[idx]
            loss_local_full[idx] = loss_local_full[idx] + copy.deepcopy(last_loss)
            
        
            # lens are the weigth of local model when doing averages
            # summing all the weights of shared global models
            if len(w_glob) == 0:
                # first iteration 
                w_glob = copy.deepcopy(w_local)
                for k,key in enumerate(net_glob.state_dict().keys()):
                    #key_full = '1.' + key
                    w_glob[key] = w_glob[key]*lens[idx]
                    w_locals[idx][key] = w_local[key]
            else:
                for k,key in enumerate(net_glob.state_dict().keys()):

                    if key in w_glob_keys:
                        w_glob[key] += w_local[key]*lens[idx]

                    w_locals[idx][key] = copy.deepcopy(w_local[key])

            times_in.append( time.time() - start_in )
            #print(times_in[-1])

        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)
        #--------------------------------------------------------------------
        # get weighted average for global weights
        # by normalized with respect to total_len
        #----------------------------------------------------------------------
        for k in net_glob.state_dict().keys():
            w_glob[k] = torch.div(w_glob[k], total_len)

        #----------------------------------------------------------------------
        # updating global model 
        # and initializing current local model with global
        #----------------------------------------------------------------------
        w_local = net_glob.state_dict()
        for k in w_glob_keys:
            w_local[k] = w_glob[k]
        if args.epochs != iter:
            net_glob.load_state_dict(w_glob)
        
        
        #----------------------------------------------------------------------
        # updating prior model 
        #----------------------------------------------------------------------
        prior.mu = prior.mu_temp/ prior.n_update
        prior.init_mu_temp()        

        if args.align_epochs_altern >0:
   
            #----------------------------------------------------------------------
            #  Reajust mapping with respect to the new means
            #----------------------------------------------------------------------
            prior.mu.requires_grad_(False)
            train_preproc(net_preprocs,user_data,prior,
                          n_epochs=args.align_epochs_altern,
                          args=args,
                          verbose=False
                          )
        #
        #    verbosity
        #
   
        if iter % args.test_freq==args.test_freq-1 or iter>=args.epochs-10:
            if times == []:
                times.append(max(times_in))
            else:
                times.append(times[-1] + max(times_in))
                
            acc_test, loss_test,bal_acc_test = het_test_img_local_all(net_glob, net_preprocs, args,
                                                         user_test_data,
                                                        w_glob_keys=w_glob_keys,
                                                        w_locals=w_locals,indd=indd)
            acc_train, loss_tr, bal_acc_train= het_test_img_local_all(net_glob, net_preprocs, args,
                                                         user_data,
                                                        w_glob_keys=w_glob_keys,
                                                        w_locals=w_locals,indd=indd)
            
            acc_train = bal_acc_train if args.dataset=='isic2019' else acc_train
            acc_test = bal_acc_test if args.dataset == 'isic2019' else acc_test

            
            accs.append(acc_test)
            accs_train.append(acc_train)

            # for algs which learn a single global model, these are the local accuracies (computed using the locally updated versions of the global model at the end of each round)
            if iter != args.epochs and iter%10 != 0:
                print('Round {:3d}, Train loss: {:.3f}, Test loss: {:.3f}, Local Test accuracy: {:.2f}'.format(
                        iter, loss_tr, loss_test, acc_test))
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

    end = time.time()
    print(end-start)
    print(times)
    print(accs)
    print(args)
    
    
    #%%
    save_dir = args.savedir
    out_dir = f"{args.dataset}-{args.num_users:d}-{args.shard_per_user:d}-frac{args.frac:2.1f}"
    out_dir += f"-upd_p{args.update_prior:}-upd_prp{args.update_net_preproc:}"
    out_dir += f"-reg_w{args.reg_w:2.3f}--reg_w{args.reg_class_prior:2.3f}/"
    
    opt = copy.deepcopy(vars(args))    
    for keys in ['full_client_freq','num_workers','device','mean_target_variance',
                  'align_epochs_altern','num_classes','subsample_client_data',
                  'gpu','test_freq','n_per_class','dataset','num_users','shard_per_user','frac',
                  'update_prior','update_net_preproc','reg_w','reg_class_prior','savedir']:
        opt.pop(keys, None)
    
    key_orig = ['local_ep','local_rep_ep','model_type']
    key_new = ['l_ep','l_repep','mdl','reg_cp']
    
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
    base_dir_lc = filename + 'Loss_Curve.csv'
    user_save_path = base_dir
    accs_m = np.array([accs,accs_train])
    times= np.array(times)

    accs_frame = pd.DataFrame({'accs_test': accs_m[0, :], 'accs_train': accs_m[1, :],'times':times})
    accs_frame.to_csv(save_dir+out_dir+base_dir, index=False)
    
    loss_curve = pd.DataFrame(loss_local_full)
    loss_curve.to_csv(save_dir + out_dir + base_dir_lc, index=False)

        
#%%
