#!/bin/bash
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=info
torchrun --nnodes=2 --nproc_per_node=1 --node_rank=0 --master_addr=192.168.1.70 --master_port=29500 unbalanced_test.py --json_path='/home/args.json'