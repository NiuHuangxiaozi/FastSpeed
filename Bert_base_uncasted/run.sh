#!/bin/bash
export NCCL_IB_DISABLE=1
torchrun --nnodes=2 --nproc_per_node=2 --node_rank=1 --master_addr=192.168.1.131 --master_port=29502 main.py --json_path='/home/args.json' > manualBertbaseuncased_4GPU_V100_Log.txt 2>&1