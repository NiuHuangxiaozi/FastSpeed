


#自己写的其他的库
from utils.sampler import Distributed_Elastic_Sampler
from debug import global_rank_print



#官方的库
import sys
import torch
from torch.utils.data import DataLoader


'''
这个文件的作用是实现数据集的合理划分以及模型流水线并行的合理划分
'''
__all__ = ["OptimusSpeed", ]





class OptimusSpeed:
    def __init__(self)->None:
        pass

    '''
    build_model构建我们需要训练的模型
    '''
    def build_model(self):
        pass



    def DDP_unbalanced_split(self):
        pass
    def dataset_split(
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
                    pass

                #for debug
                global_rank_print(0,"My global rank is %d. The partition_list is : %d" % (0,partition_list))
                global_rank_print(0,"My global rank is %d. The batchsize_list is : %d" % (0,batchsize_list))

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
                raise  TypeError
        except TypeError:
            print("Your choice of partition_method is wrong [neither manual or autobalanced!")
            sys.exit(1)
        except AttributeError:
            print("partition_list can't be None in the manual mode!")
            sys.exit(1)

