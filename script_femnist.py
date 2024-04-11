#!/usr/bin/env python3
# -*- coding: utf-8 -*-





import os
import argparse

parser = argparse.ArgumentParser()    

parser.add_argument('--s', type=int)
    
args = parser.parse_args()



reg_w = 0.001
reg_class = 0.001
n_seed=3
num_classes = 10
gpu = -1
coms = ''
k = 0
num_users = 50
shard_per_user = 1
data = 'femnist'


for setting in [args.s]:
    for seed in range(0,n_seed):
        if setting == 0:
            # FLIC Classif
            command = f"nohup python -u main_flic.py --seed {seed:} --dataset {data:} --num_users {num_users:} --model_type classif"
            command += f" --gpu {gpu:} --align_epochs 1000"
            command += f" --shard_per_user {shard_per_user} --reg_w {reg_w:2.4f} --reg_class_prior {reg_class:2.4f}"
            command += f" --local_ep 10 --local_rep_ep 1  --epochs 1000 --align_bs 100 --local_bs 100 --num_classes {num_classes:}"
            command += f" > out_femnist_classif_{num_users:}_{shard_per_user:}_{seed:}.log "
            os.system(command)
        elif setting == 1:
            # FLIC HL
            
            command = f"nohup python -u main_flic.py --seed {seed:} --dataset {data:} --num_users {num_users:} --model_type no-hlayers"
            command += f" --gpu {gpu:} --align_epochs 1000"
            command += f" --shard_per_user {shard_per_user} --reg_w {reg_w:2.4f} --reg_class_prior {reg_class:2.4f}"
            command += f" --local_ep 10 --local_rep_ep 1  --epochs 1000 --align_bs 100 --local_bs 100 --num_classes {num_classes:}"
            command += f" > out_femnist_n_hlayers_{num_users:}_{shard_per_user:}_{seed:}.log "
            os.system(command)
            
        elif setting == 4:
            ### HETFEDREP
            command = f"nohup python -u main_flic.py --seed {seed:} --dataset {data:} --num_users {num_users:} --model_type no-hlayers"
            command += f" --gpu {gpu:} --align_epochs -1 --reg_w 0 --reg_class_prior 0 --start_optimize_rep 0"
            command += f" --shard_per_user {shard_per_user} "
            command += f" --local_ep 10 --local_rep_ep 1 --epochs 1000  --align_bs 100 --local_bs 100 --num_classes {num_classes:}"
            command += f" > out_femnist_non_n_hlayers_{num_users:}_{shard_per_user:}_{seed:}.log "
            os.system(command)
        elif setting == 2:
            # FedHENN

            command = f"python -u main_het_competitor.py --seed {seed:} --alg hetarch --dataset {data:} --num_users {num_users:}"
            command += f" --gpu {gpu:}"
            command += f" --shard_per_user {shard_per_user}"
            command += f" --local_ep 10  --epochs 1000 --local_bs 100 --num_classes {num_classes:}"
            command += f" > out_femnist_hetarch_{num_users:}_{seed:}.log "
            os.system(command)
            k +=1
        elif setting == 3:
            # LOCAL
            command = f"python -u main_het_competitor.py --seed {seed:} --alg local --dataset {data:} --num_users {num_users:}"
            command += f" --gpu {gpu:}"
            command += f" --shard_per_user {shard_per_user}"
            command += f" --local_ep 10  --epochs 1000 --local_bs 100 --num_classes {num_classes:}"
            command += f" > out_femnist_local_{num_users:}_{shard_per_user:}_{seed:}.log "
            os.system(command)
            k +=1

        elif setting == 5:

            ### FedRep
            assert data == 'femnist'
            n_hidden = 64
            command = f"nohup python -u main_fedrep.py --seed {seed:} --dataset {data:} --num_users {num_users:} "
            command += f" --gpu {gpu:} --dim_hidden {n_hidden:} "
            command += f" --local_ep 10 --local_updates 50000 --epochs 1000"
            command += f" > out_femnist_fedrep_{num_users:}_{seed:}.log "
            os.system(command)