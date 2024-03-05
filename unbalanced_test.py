
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
    train_dataset,test_dataset=get_dataset(args.local_rank,param.data_path)
    #定义模型
    task_model=AlexNet()
    #进行异构的数据划分
    train_loader = optimus.ddp_unbalanced_dataset_split(
                                        global_rank=args.global_rank,
                                        dataset=train_dataset,
                                        partition_method=param.data_partition_method,
                                        partition_list=param.manual_partition_lists,
                                        batchsize_list=param.batch_size,
                                        model=task_model,
                                        train_loader_pin_memory=param.train_loader_pin_memory,
                                        train_loader_num_workers=param.train_loader_num_workers
                                        )
    #打印模型的参数量
    if args.local_rank==0:
        print("The moedl's total parameter is ",optimus.get_parameter(task_model))
        print('In task, we use {} gpus!'.format(torch.cuda.device_count()))

    wrapped_model=optimus.model_wrap(task_model, args.local_rank,param.model_parallelism)

    optimizer = optim.Adam(wrapped_model.parameters(), lr=param.learning_rate,)
    criterion = nn.CrossEntropyLoss()

    wrapped_model.train()
    for epoch in range(param.epochs):  # loop for many epochs
        running_loss = 0.0
        train_loader.sampler.set_epoch(epoch)
        optimus.time_start(args.local_rank,0)
        for i, data in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].cuda(args.local_rank), data[1].cuda(args.local_rank)
            if i == 0: #tes whether the dataloader is right or not.
                print("The input shape is ", inputs.shape)
            loss=optimus.unbalanced_train(i,
                                          param.gradient_accumulate_step,
                                          args.global_rank,
                                          wrapped_model,
                                          inputs,
                                          labels,
                                          criterion,
                                          optimizer)
            running_loss += loss.item()
            if i % param.log_interval == (param.log_interval - 1):  # print every log_interval mini-batches
                print('[RANK %d][EPOCH %d][INDEX %d] :Average loss: %.4f' % (args.global_rank, epoch + 1, i + 1, running_loss / param.log_interval))
                running_loss = 0.0
        optimus.time_end(args.local_rank, 0)
        if args.local_rank==0:
            print('local rank %d The epoch %d time cost is %.5f' % (args.local_rank, epoch, optimus.calculate_time(args.local_rank,0)))

    print('Global Rank %d finished training' % (args.global_rank))




# /////////////////////////////////////////////////////////////////
if __name__ == '__main__':
    main()