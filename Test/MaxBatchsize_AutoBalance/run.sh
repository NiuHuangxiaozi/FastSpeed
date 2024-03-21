#!/bin/bash
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export NCCL_IB_DISABLE=1
torchrun --nnodes=1 --nproc_per_node=4 --node_rank=0 --master_addr=127.0.0.1 --master_port=29500  main.py