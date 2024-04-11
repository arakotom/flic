
# Modified from: https://github.com/pliang279/LG-FedAvg/blob/master/main_fed.py
# credit goes to: Paul Pu Liang

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

# This program implements FedRep under the specification --alg fedrep, as well as Fed-Per (--alg fedper), LG-FedAvg (--alg lg), 
# FedAvg (--alg fedavg) and FedProx (--alg prox)
#%%
import copy
import itertools
import numpy as np
import pandas as pd
import torch
from torch import nn
import random
import os

#from utils.options import args_parser
from utils.train_utils import get_data, get_model, read_data
from models.Update import LocalUpdate
from models.test import test_img_local_all
import sys
import time
import argparse


if __name__ == '__main__':
    if 0 :
        sys.argv = ['']
        print("no args")
    parser = argparse.ArgumentParser()    

    parser.add_argument('--alg', type=str, default='fedrep', help="Algorithm")
    #parser.add_argument('--dataset', type=str, default='bci-subset-common', help="choice of the dataset")
    parser.add_argument('--dataset', type=str, default='femnist', help="choice of the dataset")

    parser.add_argument('--num_users', type=int, default=10, help="number of users")
    parser.add_argument('--shard_per_user', type=int, default=2, help="number of classes per user")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")

    parser.add_argument('--epochs', type=int, default=50,help="rounds of training")
    parser.add_argument('--local_bs', type=int, default=100, help="local batch size")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate for model")
    parser.add_argument('--local_ep', type=int, default=10, help="number of local epoch")
    parser.add_argument('--local_updates', type=int, default=5000, help="number of local epoch")
    parser.add_argument('--dim_hidden', type=int, default=64, help="hidden layers")
    parser.add_argument('--dim_latent', type=int, default=64, help="hidden layers")

    parser.add_argument('--local_rep_ep', type=int, default=1, help="number of local epoch for representation among local_ep")
    parser.add_argument('--reg_w', type=float, default=0.001, help="regularization of W ")
    parser.add_argument('--reg_class_prior', type=float, default=0.001, help="regularization on class of prior (\lambda_2) ")
    parser.add_argument('--model', type=str, default='mlp', help="choosing the global model, [classif, no-hlayers, 2-hlayers]")
    parser.add_argument('--num_workers', type=int, default=0, help="number of workers in dataloader")
    parser.add_argument('--test_freq', type=int, default=1, help="frequency of test eval")

    parser.add_argument('--seed', type=int, default=0,help="choice of seed")
    parser.add_argument('--gpu', type=int, default=0, help="gpu to use (if any")
    parser.add_argument('--savedir', type=str, default='./save/', help="save dire")
 
 
 


    args = parser.parse_args()

    #args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print(args)
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    lens = np.ones(args.num_users)
    if 'cifar' in args.dataset or args.dataset == 'mnist':
        dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args)
        for idx in dict_users_train.keys():
            np.random.shuffle(dict_users_train[idx])

    elif 'textcaps-clip' in args.dataset:
        from Het_data import get_textcaps_clip
        user_data, user_data_addeddim, user_test_data, user_data_addeddim  =  get_textcaps_clip(args)
    elif 'MU-resize' in args.dataset:
        from Het_data import get_mnist_usps 
        args.num_classes = 10
        args.image_size = 28      
        user_data, user_data_addeddim, user_test_data, user_data_addeddim  =  get_mnist_usps(args)
    elif args.dataset == 'bci-subset-common':
        from Het_data import get_bci_subset_common
        args.num_classes = 5
        args.num_users = 40
        user_data, user_data_addeddim, user_test_data, user_data_addeddim  = get_bci_subset_common(args)
        lens = np.ones(args.num_users)
    elif args.dataset == 'toy_2_projection':
        from Het_data import  get_toy_2_projection
        
        user_data, user_data_addeddim, user_test_data, user_data_addeddim  = get_toy_2_projection(args)
        args.dim = args.dim_latent
    elif args.dataset == 'toy_3_projection':
        from Het_data import  get_toy_3_projection
        
        user_data, user_data_addeddim, user_test_data, user_data_addeddim  = get_toy_3_projection(args)
        args.dim = args.dim_latent
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
            
        lens = np.ones(args.num_users)
    print(args.alg)
    print(args)

    

    
    # build model
    net_glob = get_model(args)
    net_glob.train()


    total_num_layers = len(net_glob.state_dict().keys())
    print(net_glob.state_dict().keys())
    net_keys = [*net_glob.state_dict().keys()]

    # specify the representation parameters (in w_glob_keys) and head parameters (all others)
    if args.alg == 'fedrep' or args.alg == 'fedper':
        if 'cifar' in  args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [0,1,3,4]]
        elif 'mnist' in args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [0,1,2]]
        elif 'textcaps' in args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [0,1,2]]
        elif 'bci' in args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [0,1,2]]
        elif args.dataset =='MU-resize':
            w_glob_keys = [net_glob.weight_keys[i] for i in [0,1,2]]
        elif 'sent140' in args.dataset:
            w_glob_keys = [net_keys[i] for i in [0,1,2,3,4,5]]
        elif 'toy_2' in args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [0,1,2]]
        elif 'toy_3' in args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [0,1,2]]
        else:
            w_glob_keys = net_keys[:-2]
    elif args.alg == 'lg':
        if 'cifar' in  args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [1,2]]
        elif 'mnist' in args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [2,3]]
        elif 'sent140' in args.dataset:
            w_glob_keys = [net_keys[i] for i in [0,6,7]]
        else:
            w_glob_keys = net_keys[total_num_layers - 2:]

    if args.alg == 'fedavg' or args.alg == 'prox' or args.alg == 'maml':
        w_glob_keys = []
    if 'sent140' not in args.dataset:
        w_glob_keys = list(itertools.chain.from_iterable(w_glob_keys))
    
    print(total_num_layers)
    print(w_glob_keys)
    print(net_keys)
    
    if args.alg == 'fedrep' or args.alg == 'fedper' or args.alg == 'lg':
        num_param_glob = 0
        num_param_local = 0
        for key in net_glob.state_dict().keys():
            num_param_local += net_glob.state_dict()[key].numel()
            print(num_param_local)
            if key in w_glob_keys:
                num_param_glob += net_glob.state_dict()[key].numel()
        percentage_param = 100 * float(num_param_glob) / num_param_local
        print('# Params: {} (local), {} (global); Percentage {:.2f} ({}/{})'.format(
            num_param_local, num_param_glob, percentage_param, num_param_glob, num_param_local))
    print("learning rate, batch size: {}, {}".format(args.lr, args.local_bs))

    # generate list of local models for each user
    net_local_list = []
    w_locals = {}
    for user in range(args.num_users):
        w_local_dict = {}
        for key in net_glob.state_dict().keys():
            w_local_dict[key] =net_glob.state_dict()[key]
        w_locals[user] = w_local_dict

    # training
    indd = None      # indices of embedding for sent140
    loss_train = []
    accs = []
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
            start_in = time.time()
            if 'sent140' in args.dataset:
                if args.epochs == iter:
                    local = LocalUpdate(args=args, dataset=dataset_train[list(dataset_train.keys())[idx][:args.m_ft]], idxs=dict_users_train, indd=indd)
                else:
                    local = LocalUpdate(args=args, dataset=dataset_train[list(dataset_train.keys())[idx][:args.m_tr]], idxs=dict_users_train, indd=indd)
            else:
                if args.epochs == iter:
                    #local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_train[idx][:args.m_ft])
                    local = LocalUpdate(args=args, dataset= user_data[idx])

                else:
                    #local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_train[idx][:args.m_tr])
                    local = LocalUpdate(args=args, dataset= user_data[idx])

            net_local = copy.deepcopy(net_glob)
            w_local = net_local.state_dict()
            if args.alg != 'fedavg' and args.alg != 'prox':
                for k in w_locals[idx].keys():
                    if k not in w_glob_keys:
                        w_local[k] = w_locals[idx][k]
            net_local.load_state_dict(w_local)
            last = iter == args.epochs
            if 'sent140' in args.dataset:
                w_local, loss, indd = local.train(net=net_local.to(args.device),ind=idx, idx=clients[idx], w_glob_keys=w_glob_keys, lr=args.lr,last=last)
            else:
                w_local, loss, indd = local.train(net=net_local.to(args.device), idx=idx, w_glob_keys=w_glob_keys, lr=args.lr, last=last)
            loss_locals.append(copy.deepcopy(loss))
            total_len += lens[idx]
            if len(w_glob) == 0:
                w_glob = copy.deepcopy(w_local)
                for k,key in enumerate(net_glob.state_dict().keys()):
                    w_glob[key] = w_glob[key]*lens[idx]
                    w_locals[idx][key] = w_local[key]
            else:
                for k,key in enumerate(net_glob.state_dict().keys()):
                    if key in w_glob_keys:
                        w_glob[key] += w_local[key]*lens[idx]
                    else:
                        w_glob[key] += w_local[key]*lens[idx]
                    w_locals[idx][key] = w_local[key]

            times_in.append( time.time() - start_in )
        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)

        # get weighted average for global weights
        for k in net_glob.state_dict().keys():
            w_glob[k] = torch.div(w_glob[k], total_len)

        w_local = net_glob.state_dict()
        for k in w_glob.keys():
            w_local[k] = w_glob[k]
        if args.epochs != iter:
            net_glob.load_state_dict(w_glob)

        if iter % args.test_freq==args.test_freq-1 or iter>=args.epochs-10:
            if times == []:
                times.append(max(times_in))
            else:
                times.append(times[-1] + max(times_in))

            acc_test, loss_test = test_img_local_all(net_glob.to(args.device), args, user_test_data,
                                                        w_glob_keys=w_glob_keys, w_locals=w_locals,indd=indd)
            accs.append(acc_test)
            # for algs which learn a single global model, these are the local accuracies (computed using the locally updated versions of the global model at the end of each round)
            if iter != args.epochs:
                print('Round {:3d}, Train loss: {:.3f}, Test loss: {:.3f}, Test accuracy: {:.2f}'.format(
                        iter, loss_avg, loss_test, acc_test))
            else:
                # in the final round, we sample all users, and for the algs which learn a single global model, we fine-tune the head for 10 local epochs for fair comparison with FedRep
                print('Final Round, Train loss: {:.3f}, Test loss: {:.3f}, Test accuracy: {:.2f}'.format(
                        loss_avg, loss_test, acc_test))
            if iter >= args.epochs-10 and iter != args.epochs:
                accs10 += acc_test/10

            # below prints the global accuracy of the single global model for the relevant algs
            if args.alg == 'fedavg' or args.alg == 'prox':
                acc_test, loss_test = test_img_local_all(net_glob, args, dataset_test, dict_users_test,
                                                        w_locals=None,indd=indd,dataset_train=dataset_train, dict_users_train=dict_users_train, return_all=False)
                if iter != args.epochs:
                    print('Round {:3d}, Global train loss: {:.3f}, Global test loss: {:.3f}, Global test accuracy: {:.2f}'.format(
                        iter, loss_avg, loss_test, acc_test))
                else:
                    print('Final Round, Global train loss: {:.3f}, Global test loss: {:.3f}, Global test accuracy: {:.2f}'.format(
                        loss_avg, loss_test, acc_test))
            if iter >= args.epochs-10 and iter != args.epochs:
                accs10_glob += acc_test/10

        # if iter % args.save_every==args.save_every-1:
        #     model_save_path = './save/accs_'+ args.alg + '_' + args.dataset + '_' + str(args.num_users) +'_'+ str(args.shard_per_user) +'_iter' + str(iter)+ '.pt'
        #     torch.save(net_glob.state_dict(), model_save_path)

    print('Average accuracy final 10 rounds: {}'.format(accs10))
    if args.alg == 'fedavg' or args.alg == 'prox':
        print('Average global accuracy final 10 rounds: {}'.format(accs10_glob))
    end = time.time()
    print(end-start)
    print(times)
    print(accs)
    base_dir = './save/accs_' + args.alg + '_' +  args.dataset + str(args.num_users) +'_'+ str(args.shard_per_user) + '.csv'
    user_save_path = base_dir
    accs = np.array(accs)
    accs = pd.DataFrame(accs, columns=['accs'])
    accs.to_csv(base_dir, index=False)

print(args)


save_dir = args.savedir
out_dir = f"{args.dataset}-{args.num_users:d}-{args.shard_per_user:d}-frac{args.frac:2.1f}-fedrep/"

# for toy in tmlr

if 'toy_2' in args.dataset:
    out_dir = f"toy_2-{args.num_users:d}-{args.shard_per_user:d}-frac{args.frac:2.1f}/"
elif 'toy_3' in args.dataset:
    out_dir = f"toy_3-{args.num_users:d}-{args.shard_per_user:d}-frac{args.frac:2.1f}/"


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
accs_m = np.array(accs).reshape(-1)
times= np.array(times)

accs_frame = pd.DataFrame({'accs_test': accs_m,'times':times})
accs_frame.to_csv(save_dir+out_dir+base_dir, index=False)




# %%
