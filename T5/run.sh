#!/bin/bash
export NCCL_IB_DISABLE=1
/home/ainet/anaconda3/envs/deepspeed/bin/torchrun --nnodes=1 --nproc_per_node=2 --node_rank=0 --master_addr=127.0.0.1 --master_port=29502 main.py --json_path='/home/ainet/wsj/T5/args.json'