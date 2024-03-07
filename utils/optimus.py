
# 自己写的其他的库
from utils.sampler import Distributed_Elastic_Sampler
from help.debug import global_rank_print
from utils.calculation import calculate_time, calculate_param
# exception
from help.debug import MyException

# 官方的库
import copy
import sys
import torch
import time
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import List ,Union
import torch.distributed as dist

'''
这个文件的作用是实现数据集的合理划分以及模型流水线并行的合理划分
'''
__all__ = ["OptimusSpeed", ]


class OptimusSpeed:
    def __init__(self ) ->None:
        self.grad_portion =None  # 对于数据并行不平衡的batch的normalize
        # 用来计算模型的运行时间并做一些分析
        self.start_time = 0.0
        self.end_time = 0.0

    '''
    build_model构建我们需要训练的模型
    '''

    def model_wrap(self, model, device, model_type: str):
        try:
            if model_type == 'ddp':
                self.wrapped_model = DDP(model.to(device))
            elif model_type == 'pipe':
                pass
            elif model_type == 'hybrid':
                pass
            else:
                raise ValueError
        except ValueError:
            print("Current model parallel type only contain ddp, pipe and hybrid(ddp,pipe)")
            sys.exit(1)
        return self.wrapped_model

    def autobalance(self,
                    model,
                    world_size,
                    criterion,
                    config_param,
                    args
                    ) -> Union[List[int], List[int]]:
        partition_list = None
        batchsize_list = None
        try:
            if config_param.model_type == 'ddp':
                dummy_model = copy.deepcopy(model)  # 独立拷贝一份模型用作测试防止修改原来的模型
                if config_param.unbalanced_strategy == "time":

                    dummy_time=calculate_time(dummy_model,criterion,args.local_rank,config_param.data_dummy_shape)
                    dummy_time_tensor = torch.tensor(dummy_time).to(args.local_rank)
                    time_list = [torch.zeros(1, ).to(args.local_rank) for i in range(0, world_size)]
                    print("time_list",time_list)
                    print("dummy_time_tensor",dummy_time_tensor)
                    dist.all_gather(time_list, dummy_time_tensor)
                    print("time_list",time_list)
                    time_list = [obj.item()*100 for obj in time_list]
                    rounded_time_array=np.trunc(time_list)/100
                    #rounded_time_array = (np.rint(time_list)).astype(int)
                    print("rounded_time_array",rounded_time_array)
                    portion = rounded_time_array / np.sum(rounded_time_array)
                    print("portion",portion)
                    partition_list = (config_param.total_datasize * portion).astype(int)
                    batchsize_list = (config_param.batch_size * portion).astype(int)
                    partition_list[-1]=config_param.total_datasize-np.sum(partition_list[:-1])
                    batchsize_list[-1]=config_param.batch_size-np.sum(batchsize_list[:-1])
                    return partition_list.tolist(), batchsize_list.tolist()
                else:
                    raise MyException(2, "The auto balanced strategy is only support for time and complexity now.")
            elif config_param.model_type == 'pipe':
                pass
            else:
                raise MyException(3, "The parallel type only support for ddp ,pipe or hybrid")
        except MyException as e:
            print(e)
            sys.exit(1)

    def ddp_unbalanced_dataset_split(
            self,
            global_rank,
            dataset,
            partition_list,
            batchsize_list,
            model,
            criterion,
            world_size,
            config_param,
            args
    ) -> torch.utils.data.DataLoader:
        try:
            if config_param.partition_method == 'manual' or config_param.partition_method == 'autobalanced':
                if config_param.partition_method == 'autobalanced':
                    print("This is autobalanced partition mode!")
                    partition_list, batchsize_list = self.autobalance(model=model, world_size=world_size,
                                                                      criterion=criterion, config_param=config_param,
                                                                      args=args)

                # for debug
                global_rank_print(0, "The partition_list is : " + str(partition_list))
                global_rank_print(0, "The batchsize_list is : " + str(batchsize_list))
                if partition_list is None or batchsize_list is None:
                    raise MyException(0, "partition_list or batchsize_list can't be None in the manual mode!")
                # 这里在计算完batchsize_list之后我们需要计算一下各个梯度的权重比例
                self.grad_portion = batchsize_list / np.sum(batchsize_list)

                sampler_dict = \
                    {
                        'method': "uneven",
                        'partition_list': partition_list
                    }

                sampler = Distributed_Elastic_Sampler(dataset=dataset, partition_strategy=sampler_dict)
                train_loader = DataLoader(dataset=dataset,
                                          batch_size=batchsize_list[global_rank],
                                          shuffle=False,  # 这个值必须设置为false，否则会导致多个节点可能都抽到同一个样例的结果
                                          sampler=sampler,
                                          pin_memory=config_param.train_loader_pin_memory,
                                          num_workers=config_param.train_loader_num_workers
                                          )
                return train_loader
            else:
                raise MyException(1, "Your choice of partition_method(" + str(
                    config_param.partition_method) + ") is wrong [neither manual or autobalanced!")
        except MyException as e:
            print(e)
            sys.exit(1)

    def unbalanced_train(self,
                         iter_index,
                         gradient_accumulate_step,
                         global_rank,
                         wrapped_model,
                         inputs,
                         labels,
                         criterion,
                         optimizer) -> torch.Tensor:
        # 前向传播
        outputs = wrapped_model(inputs)
        # 计算loss值
        loss = criterion(outputs, labels)
        (loss * self.grad_portion[global_rank]).backward()

        if (iter_index + 1) % gradient_accumulate_step == 0:
            # 4.1 update parameters of net
            optimizer.step()
            # 4.2 reset gradient
            optimizer.zero_grad()
        return loss

    @staticmethod
    def get_parameter(model) -> dict:
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    def time_start(self, local_rank, target_local_rank):
        if local_rank == target_local_rank:
            self.start_time = time.time()

    def time_end(self, local_rank, target_local_rank):
        if local_rank == target_local_rank:
            self.end_time = time.time()

    def calculate_time(self, local_rank, target_local_rank):
        if local_rank == target_local_rank:
            return self.end_time - self.start_time






