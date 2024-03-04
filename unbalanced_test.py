
# official lib
from torch.distributed import init_process_group, destroy_process_group
import torch
import os
import argparse
from argparse import Namespace
# my lib
from  utils.load_json import Params

########################################################################################################################


def Get_args():
    parser = argparse.ArgumentParser(description='Alexnet train on cifar10.')
    parser.add_argument('--json_path', default="./args.json", help="args.json file path")
    parser.add_argument('--local_rank', default=-1, type=int, help='Local rank always refer to specific gpu.')
    parser.add_argument('--global_rank', default=-1, type=int, help='Global Rank.')

    args = parser.parse_args()
    #获得分布式训练的本地rank和全局的rank。
    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.global_rank = int(os.environ["RANK"])
    return args

def Distributed_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
def Distributed_destroy():
    destroy_process_group()

def main():
    Distributed_setup()
    args = Get_args()
    param = Params(args.json_path)
    if args.local_rank == 0:
        print("The config is :", vars(param))
    Task(args,param)
    print("RANK :", args.global_rank, "All Finished")
    Distributed_destroy()

def Task(
        args:Namespace,
        param:Params
         ):

    print("In task!")




# /////////////////////////////////////////////////////////////////
if __name__ == '__main__':
    main()