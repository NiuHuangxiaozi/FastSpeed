#!/bin/bash
export NCCL_IB_DISABLE=1
torchrun --nnodes=1 --nproc_per_node=4 --node_rank=0 --master_addr=127.0.0.1 --master_port=29502 main.py --json_path='/home/args.json' > Virtual_Bert_4GPU_2A162[A16]_ManualB30Log.txt 2>&1