import torch
import os
import torch.distributed as dist
from torch.distributed import init_process_group,destroy_process_group
from sampler import *
from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader,Dataset
from simpleDataset import DummyDataset


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




def TestCifar10(local_rank, global_rank, world_size):
    train_dataset, test_dataset = get_dataset(local_rank, "/home/")
    partition_list = [30000, 20000]
    batchsize_list = [30, 20]
    sampler_dict = \
        {
            'method': "uneven",
            'partition_list': partition_list
        }

    sampler = Distributed_Elastic_Sampler(dataset=train_dataset, partition_strategy=sampler_dict)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batchsize_list[global_rank],
                              shuffle=False,  # 这个值必须设置为false，否则会导致多个节点可能都抽到同一个样例的结果
                              sampler=sampler,
                              pin_memory=True,
                              num_workers=4
                              )

    for index, value in enumerate(train_loader):
        if index == 0:
            data, label = value
            print(f"The rank is {global_rank}. And I get the data shape is {data.shape} label shape is {label.shape}!")
        else:
            break
    print("test over")


def TestDummyDataset(local_rank, global_rank, world_size):
    dataset=DummyDataset()
    partition_list = [5,10,15,20]
    batchsize_list = [1,2,3,4]
    sampler_dict = \
        {
            'method': "uneven",
            'partition_list': partition_list
        }

    sampler = Distributed_Elastic_Sampler(dataset=dataset,shuffle=True, partition_strategy=sampler_dict)
    train_loader = DataLoader(dataset=dataset,
                              batch_size=batchsize_list[global_rank],
                              shuffle=False,  # 这个值必须设置为false，否则会导致多个节点可能都抽到同一个样例的结果
                              sampler=sampler,
                              pin_memory=True,
                              num_workers=4
                              )
    for epoch in range(2):
        train_loader.sampler.set_epoch(epoch)
        train_iter=iter(train_loader)
        i=0
        print(i)
        while i<5:
                value=next(train_iter)
                data, label = value
                print(f"Epoch {epoch} Iter {i}:->The rank is {global_rank}. And I get the data is {data} label shape is {label}!")
                i=i+1
    print("test over")


def Task(local_rank, global_rank, world_size):
    # TestCifar10(local_rank, global_rank, world_size)
    TestDummyDataset(local_rank, global_rank, world_size)


def Distributed_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def Distributed_destroy():
    destroy_process_group()

def main():
    Distributed_setup()
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = dist.get_world_size()
    Task(local_rank, global_rank, world_size)
    print("RANK :", global_rank, "All Finished")
    Distributed_destroy()




if __name__=="__main__":
    main()