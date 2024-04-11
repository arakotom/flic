# Modified from: https://github.com/pliang279/LG-FedAvg/blob/master/models/test.py
# credit goes to: Paul Pu Liang

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import copy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import time
from models.language_utils import get_word_emb_arr, repackage_hidden, process_x, process_y

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        d = int(self.idxs[item])
        image, label = self.dataset[d]
        return image, label

class DatasetSplit_leaf(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[item]
        return image, label

def test_img_local(net_g, user_data, args,idx=None,indd=None, user_idx=-1, idxs=None):
    net_g.eval()
    test_loss = 0
    correct = 0

    # put LEAF data into proper format
    # if 'femnist' in args.dataset:
    #     leaf=True
    #     datatest_new = []
    #     usr = idx
    #     for j in range(len(dataset[usr]['x'])):
    #         datatest_new.append((torch.reshape(torch.tensor(dataset[idx]['x'][j]),(1,28,28)),torch.tensor(dataset[idx]['y'][j])))
    # elif 'sent140' in args.dataset:
    #     leaf=True
    #     datatest_new = []
    #     for j in range(len(dataset[idx]['x'])):
    #         datatest_new.append((dataset[idx]['x'][j],dataset[idx]['y'][j]))
    # else:
    #     leaf=False
    
    # if leaf:
    #     data_loader = DataLoader(DatasetSplit_leaf(datatest_new,np.ones(len(datatest_new))), batch_size=args.local_bs, shuffle=False)
    # else:
    #     data_loader = DataLoader(DatasetSplit(dataset,idxs), batch_size=args.local_bs,shuffle=False)
    # if 'sent140' in args.dataset:
    #     hidden_train = net_g.init_hidden(args.local_bs)

    count = 0
    data_loader = DataLoader(user_data, batch_size=200, shuffle=True,drop_last=False)
    for idx, (data, target) in enumerate(data_loader):
        if 'sent140' in args.dataset:
            input_data, target_data = process_x(data, indd), process_y(target, indd)
            if args.local_bs != 1 and input_data.shape[0] != args.local_bs:
                break

            data, targets = torch.from_numpy(input_data).to(args.device), torch.from_numpy(target_data).to(args.device)
            net_g.zero_grad()

            hidden_train = repackage_hidden(hidden_train)
            output, hidden_train = net_g(data, hidden_train)

            loss = F.cross_entropy(output.t(), torch.max(targets, 1)[1])
            _, pred_label = torch.max(output.t(), 1)
            correct += (pred_label == torch.max(targets, 1)[1]).sum().item()
            count += args.local_bs
            test_loss += loss.item()

        else:
            if args.gpu != -1:
                data, target = data.to(args.device), target.to(args.device)
            log_probs = net_g(data)
            # sum up batch loss
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    if 'sent140' not in args.dataset:
        count = len(data_loader.dataset)
    test_loss /= count
    accuracy = 100.00 * float(correct) / count
    return  accuracy, test_loss

def test_img_local_all(net, args, users_test_data ,w_locals=None,w_glob_keys=None, indd=None,dataset_train=None,dict_users_train=None, return_all=False):

    
    tot = 0
    num_idxxs = args.num_users
    acc_test_local = np.zeros(num_idxxs)
    loss_test_local = np.zeros(num_idxxs)
    for idx in range(num_idxxs):
        # net is the global network.
        # it is copied and then local layers are overwritten.
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
        if 'sent140' in args.dataset:
            a, b =  test_img_local(net_local, dataset_test, args,idx=dict_users_test[idx],indd=indd, user_idx=idx)
            n_test = len(dataset_test[dict_users_test[idx]]['x']) 
            tot += n_test
        elif args.dataset == 'MU-resize':
            a, b = test_img_local(net_local,  users_test_data[idx], args, user_idx=idx) 
            tot +=  len(users_test_data[idx])
            n_test = len(users_test_data[idx])
        else:
            a, b = test_img_local(net_local,  users_test_data[idx], args, user_idx=idx) 
            tot +=  users_test_data[idx].tensors[0].shape[0]
            n_test = users_test_data[idx].tensors[0].shape[0]
        if  'sent140' in args.dataset:
            acc_test_local[idx] = a*len(dataset_test[dict_users_test[idx]]['x'])
            loss_test_local[idx] = b*len(dataset_test[dict_users_test[idx]]['x'])
        else:
            acc_test_local[idx] = a*n_test
            loss_test_local[idx] = b*n_test
        del net_local
    
    if return_all:
        return acc_test_local, loss_test_local
    return  sum(acc_test_local)/tot, sum(loss_test_local)/tot