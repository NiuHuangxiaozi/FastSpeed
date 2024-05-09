#!/bin/bash
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export NCCL_IB_DISABLE=1
torchrun --nnodes=1 --nproc_per_node=2 --node_rank=0 --master_addr=127.0.0.1 --master_port=29502 main_vgg.py --json_path='./args.json' >Virtual_Vgg19_cifar10_2GPU_A16[A16]_B24log.txt 2>&1