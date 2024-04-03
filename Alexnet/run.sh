#!/bin/bash
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export NCCL_IB_DISABLE=1
torchrun --nnodes=2 --nproc_per_node=2 --node_rank=0 --master_addr=192.168.1.118 --master_port=29502 main_alexnet.py --json_path='/home/args.json' > autoAlexnet_Cifar10_4GPU_A16_Log.txt 2>&1