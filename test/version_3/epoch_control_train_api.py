
# official lib
from torch.distributed import init_process_group, destroy_process_group
import torch
import os
import argparse
from argparse import Namespace
import torchvision.transforms as transforms
import torch.distributed as dist
import torchvision
import torch.optim as optim
import torch.nn as nn
# my lib
from  utils.load_json import Params
from utils.optimus import OptimusSpeed
from model import AlexNet
########################################################################################################################
def Get_args():
    parser = argparse.ArgumentParser(description='Alexnet train on cifar10.')
    parser.add_argument('--json_path', default="./args.json", help="args.json file path")
    parser.add_argument('--local_rank', default=-1, type=int, help='Local rank always refer to specific gpu.')
    parser.add_argument('--global_rank', default=-1, type=int, help='Global Rank.')
    parser.add_argument('--world_size', default=-1, type=int, help='All ranks.')
    args = parser.parse_args()
    #获得分布式训练的本地rank和全局的rank。
    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.global_rank = int(os.environ["RANK"])
    args.world_size = int(os.environ['WORLD_SIZE'])
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

def get_dataset(local_rank, dl_path):
    #图片的数据处理
    transform = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(449),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
    #保证所有的进程都到这一步
    dist.barrier()
    #所有非零local_rank都原地等待
    if local_rank != 0:
        dist.barrier()
    #加载数据集
    trainset = torchvision.datasets.CIFAR10(root=dl_path,
                                            train=True,
                                            download=True,
                                            transform=transform)
    testset = torchvision.datasets.CIFAR10(root=dl_path,
                                           train=False,
                                           download=True,
                                           transform=transform)
    #让local_rank 0 与其他的rank进行同步
    if local_rank == 0:
        dist.barrier()
    return trainset, testset

def Task(
        args:Namespace,
        param:Params
         ):
    #定义异构训练接口
    optimus=OptimusSpeed()
    #获取数据集
    train_dataset,test_dataset=get_dataset(args.local_rank,
                                           param.data_path)
    #定义模型
    task_model=AlexNet()
    #定义loss函数
    criterion = nn.CrossEntropyLoss()
    #进行异构的数据划分
    train_loader = optimus.ddp_unbalanced_dataset_split(
                                        dataset=train_dataset,
                                        model=task_model,
                                        criterion=criterion,
                                        config_param=param,
                                        args=args)

    if args.local_rank==0:
        print("The model's total parameter is ",optimus.get_parameter(task_model))
        print('In task on this node, we use {} gpus!'.format(torch.cuda.device_count()))

    wrapped_model = optimus.model_wrap(task_model,
                                       args.local_rank,
                                       param.model_type)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(wrapped_model.parameters(), lr=param.learning_rate, )


    trained_model,epoch_loss_list=optimus.train(train_loader=train_loader,
                                                wrapped_model=wrapped_model,
                                                criterion=criterion,
                                                optimizer=optimizer,
                                                param=param,
                                                args=args)

    print("epoch_loss_list is",epoch_loss_list)
    print('Global Rank %d finished training' % (args.global_rank))

# /////////////////////////////////////////////////////////////////
if __name__ == '__main__':
    main()



