#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


"""




import os


list_setting = [0]

reg_w = 0.001
reg_class = 0.001
n_seed=3
num_classes = 5
gpu = -1
coms = ''
k = 0
num_users = 54
shard_per_user = 1
data = 'bci-subset-common'
#data = 'bci-subset-specific'
#data = 'bci-full'

for setting in list_setting:
    for seed in range(0,n_seed):
        if setting == 0:
            # FLIC Classif
            command = f"nohup python -u main_flic.py --seed {seed:} --dataset {data:} --num_users {num_users:} --model_type classif"
            command += f" --gpu {gpu:} --align_epochs 100"
            command += f" --shard_per_user {shard_per_user} --reg_w {reg_w:2.4f} --reg_class_prior {reg_class:2.4f}"
            command += f" --local_ep 10 --local_rep_ep 1  --epochs 50 --align_bs 100 --local_bs 100 --num_classes {num_classes:}"
            command += f" > out_bci_classif_{num_users:}_{shard_per_user:}_{seed:}.log "
            os.system(command)
        if setting == 1:
            # FLIC HL

            
            command = f"nohup python -u main_flic.py --seed {seed:} --dataset {data:} --num_users {num_users:} --model_type no-hlayers"
            command += f" --gpu {gpu:}"
            command += f" --shard_per_user {shard_per_user} --reg_w {reg_w:2.4f} --reg_class_prior {reg_class:2.4f}"
            command += f" --local_ep 10 --local_rep_ep 1  --epochs 50 --align_bs 100 --local_bs 100 --num_classes {num_classes:}"
            command += f" > out_bci_n_hlayers_{num_users:}_{shard_per_user:}_{seed:}.log "
            os.system(command)
        elif setting == 2:
            # FedHENN
            
            command = f"python -u main_het_competitor.py --seed {seed:} --alg hetarch --dataset {data:} --num_users {num_users:}"
            command += f" --gpu {gpu:}"
            command += f" --shard_per_user {shard_per_user}"
            command += f" --local_ep 10  --epochs 50 --local_bs 100 --num_classes {num_classes:}"
            command += f" > out_bci_hetarch_{num_users:}_{seed:}.log "
            os.system(command)
            k +=1
        if setting == 3:
            # LOCAL

            command = f"python -u main_het_competitor.py --seed {seed:} --alg local --dataset {data:} --num_users {num_users:}"
            command += f" --gpu {gpu:}"
            command += f" --shard_per_user {shard_per_user}"
            command += f" --local_ep 10  --epochs 50 --local_bs 100 --num_classes {num_classes:}"
            command += f" > out_bci_local_{num_users:}_{shard_per_user:}_{seed:}.log "
            os.system(command)
            k +=1
        elif setting == 4:
            ### HETFEDREP
            command = f"nohup python -u main_flic.py --seed {seed:} --dataset {data:} --num_users {num_users:} --model_type no-hlayers"
            command += f" --gpu {gpu:} --align_epochs -1 --reg_w 0 --reg_class_prior 0 --start_optimize_rep 0"
            command += f" --shard_per_user {shard_per_user} "
            command += f" --local_ep 10 --epochs 50 --local_rep_ep 1 --local_bs 100 --num_classes {num_classes:}"
            command += f" > out_bci_bl_n_hlayers_{num_users:}_{shard_per_user:}_{seed:}.log "
            os.system(command)
        elif setting == 5:

            ### FedRep for same dimensionality
            assert data == 'bci-subset-common'
            n_hidden = 64
            command = f"nohup python -u main_fedrep.py --seed {seed:} --dataset {data:} --num_users {num_users:} "
            command += f" --gpu {gpu:} --dim_hidden {n_hidden:} "
            command += f" --shard_per_user {shard_per_user} "
            command += f" --local_ep 10 --local_updates 5000 --epochs 50"
            command += f" > out_bci_fedrep_{num_users:}_{shard_per_user:}_{seed:}.log "
            os.system(command)
