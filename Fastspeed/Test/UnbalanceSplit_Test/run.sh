#!/bin/bash
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export NCCL_IB_DISABLE=1
torchrun --nnodes=2 --nproc_per_node=2 --node_rank=1 --master_addr=192.168.1.104 --master_port=29502 main.py