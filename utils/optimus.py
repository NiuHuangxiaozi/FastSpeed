


#自己写的其他的库
from utils.sampler import Distributed_Elastic_Sampler
from help.debug import global_rank_print

#exception
from help.debug import MyException

#官方的库
import sys
import torch
import time
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

'''
这个文件的作用是实现数据集的合理划分以及模型流水线并行的合理划分
'''
__all__ = ["OptimusSpeed", ]


class OptimusSpeed:
    def __init__(self)->None:
        self.grad_portion =None #对于数据并行不平衡的batch的normalize
        #用来计算模型的运行时间并做一些分析
        self.start_time=0.0
        self.end_time=0.0

    '''
    build_model构建我们需要训练的模型
    '''
    def model_wrap(self,model,device,model_type:str):
        try:
            if model_type == 'ddp':
                self.wrapped_model = DDP(model.to(device))
            elif model_type == 'pipe':
                pass
            elif model_type=='hybrid':
                pass
            else:
                raise  ValueError
        except ValueError:
            print("Current model parallel type only contain ddp, pipe and hybrid(ddp,pipe)")
            sys.exit(1)
        return self.wrapped_model

    def ddp_unbalanced_dataset_split(
        self,
        global_rank,
        dataset,
        partition_method:str,   # manual or autobalanced，
        #manual
        partition_list,
        batchsize_list,
        #autobalanced
        model,
        train_loader_pin_memory:bool,
        train_loader_num_workers:int
    ) -> torch.utils.data.DataLoader:
        try:
            if partition_method == 'manual' or  partition_method == 'autobalanced':
                if partition_method == 'autobalanced':
                    'TODO'
                    print("This is partition mode!")
                    pass
                #for debug
                global_rank_print(0,"My global rank is %d. The partition_list is : " % (0)+str(partition_list))
                global_rank_print(0,"My global rank is %d. The batchsize_list is : " % (0)+str(batchsize_list))
                if partition_list is None or batchsize_list is None:
                    raise MyException(0, "partition_list or batchsize_list can't be None in the manual mode!")
                #这里在计算完batchsize_list之后我们需要计算一下各个梯度的权重比例
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
                                          pin_memory=train_loader_pin_memory,
                                          num_workers=train_loader_num_workers
                                          )
                return train_loader
            else:
                raise MyException(1,"Your choice of partition_method("+str(partition_method)+") is wrong [neither manual or autobalanced!")
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
                         optimizer)->torch.Tensor:
        #前向传播
        outputs = wrapped_model(inputs)
        #计算loss值
        loss = criterion(outputs, labels)
        (loss * self.grad_portion[global_rank]).backward()

        if (iter_index+ 1) % gradient_accumulate_step == 0:
            # 4.1 update parameters of net
            optimizer.step()
            # 4.2 reset gradient
            optimizer.zero_grad()
        return loss

    @staticmethod
    def get_parameter(model)->dict:
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    def time_start(self,local_rank,target_local_rank):
        if local_rank==target_local_rank:
            self.start_time = time.time()
    def time_end(self,local_rank,target_local_rank):
        if local_rank == target_local_rank:
            self.end_time=time.time()
    def calculate_time(self,local_rank,target_local_rank):
        if local_rank == target_local_rank:
            return self.end_time-self.start_time






