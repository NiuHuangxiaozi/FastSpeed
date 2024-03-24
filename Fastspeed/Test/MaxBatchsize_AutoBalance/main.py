import sys

import torch
import torch.distributed as dist
import heapq
import copy
import os
from torch.distributed import init_process_group,destroy_process_group
from typing import List

from transformers import BertTokenizer
#add my library
from search_max_batchsize import *
from Data.load_data import split_data,load_data,EmotionDataset
from model import BertClassifier,AlexNet
GREEN = '\033[92m'
END_COLOR = '\033[0m'

class MyException(Exception):
    def __init__(self, code, message):
        self.code = code
        self.message = message

    def __str__(self):
        return f"Error {self.code}: {self.message}"

def shift_load(free_devices,
               omm_devices,
               final_batch_list,
               demand_batch,
               peak_device,
               left_batch,
               valley_device):
    if demand_batch<left_batch:
        heapq.heappush(free_devices,(-(left_batch-demand_batch),valley_device))
        final_batch_list[valley_device] += demand_batch
        final_batch_list[peak_device]   -= demand_batch
    elif demand_batch==left_batch:
        final_batch_list[valley_device] += demand_batch
        final_batch_list[peak_device] -= demand_batch
    else:
        heapq.heappush(omm_devices, (-(demand_batch-left_batch),peak_device))
        final_batch_list[valley_device] += left_batch
        final_batch_list[peak_device] -= left_batch

    return free_devices,omm_devices,final_batch_list

def print_red_symbol(s:str)->None:
    print("\033[0;31;40m",s,"\033[0m")
def print_current_batchconfig(initial_list,max_batch_list):
    initial_config={
        'GPU '+str(index)+" : "+str(initial_list[index])  for index in range(len(initial_list))
    }
    maxbatch_config = {
        'GPU ' + str(index) + " : " + str(max_batch_list[index]) for index in range(len(max_batch_list))
    }
    print_red_symbol("The best batchsize config is %s"%(str(initial_config)))
    print_red_symbol("The max batchsize limit is %s" % (str(maxbatch_config)))

def batchsize_scheduling(global_rank,local_rank,initial_batch_list,max_batchsize_list)->List[int]:
        if global_rank == 0:
            assert len(initial_batch_list)==len(max_batchsize_list)
            final_batch_list=copy.deepcopy(initial_batch_list)

            try:
                free_devices=[]
                omm_devices =[]
                for index in range(len(final_batch_list)):
                    if final_batch_list[index]>max_batchsize_list[index]:
                        heapq.heappush(omm_devices, (-(final_batch_list[index]-max_batchsize_list[index]),index))
                    elif final_batch_list[index]<max_batchsize_list[index]:
                        heapq.heappush(free_devices, (-(max_batchsize_list[index]-final_batch_list[index]) , index))


                while (len(free_devices) != 0) and (len(omm_devices)!=0):
                    demand_batch,peak_device=heapq.heappop(omm_devices)
                    left_batch,valley_device=heapq.heappop(free_devices)
                    free_devices,omm_devices,final_batch_list=shift_load(free_devices,omm_devices,
                                                                         final_batch_list,
                                                                         -demand_batch,peak_device,
                                                                         -left_batch,valley_device)
                if len(omm_devices)!=0 and global_rank==0:
                    print_current_batchconfig(initial_batch_list,max_batchsize_list)
                    raise MyException(0,"Total batchsize %d can't be train on the system."%(sum(initial_batch_list)))
                final_batch_list=torch.IntTensor(final_batch_list).to(local_rank)
                dist.broadcast(tensor=final_batch_list, src=0)
                print(GREEN,'After broadcast, Rank ', global_rank, ' has data ', str(final_batch_list),END_COLOR)
            except MyException as e:
                print_red_symbol(e)
                sys.exit(1)
        else:
            final_batch_list=torch.IntTensor(len(initial_batch_list)).to(local_rank)
            dist.broadcast(tensor=final_batch_list, src=0)
            if global_rank==1:
                print(GREEN,'After broadcast, Rank ', global_rank, ' has data ',str(final_batch_list),END_COLOR)



def Distributed_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
def Distributed_destroy():
    destroy_process_group()

def main():
    Distributed_setup()
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size=dist.get_world_size()
    Task(local_rank,global_rank,world_size)
    print("RANK :", global_rank, "All Finished")
    Distributed_destroy()




def Catch_MaxBatchsize(model,
                       optimizer_name:str,
                       criterion_name:str,
                       device,
                       input_label:dict,
                       Total_Batchsize:int,
                       global_rank:int,
                       local_rank:int,
                       world_size:int
                     )->List[int]:
    # 0号节点使用gather函数
    print("Begin to test max batchsize.")

    test_platform = Test_max_batchsize(model, optimizer_name, criterion_name, device)

    test_platform.setInputData(input_label['input'])
    test_platform.setLabel(input_label['label'])

    max_batchsize = test_platform.search_max_batchsize(Total_Batchsize)
    print("The max batchsize is ", max_batchsize)
    # gather_list用于收集所有的max_batchsize
    MaxBatchsize_Tensor = torch.IntTensor([max_batchsize]).to(local_rank)
    if global_rank == 0:
        gather_list = [torch.IntTensor([1]).to(local_rank) for _ in range(world_size)]
        dist.gather(MaxBatchsize_Tensor, dst=0, gather_list=gather_list)
        print("All the max batchsize is ", gather_list)
        max_batchsize_list = [tensor_val.item() for tensor_val in gather_list]
    else:
        max_batchsize_list = [0 for _ in range(world_size)]
        dist.gather(MaxBatchsize_Tensor, dst=0)
    return max_batchsize_list


def Task(local_rank,global_rank,world_size):
    initial_batch_list = [5, 10, 15, 20]


    # #prepare for bert_base_uncased
    # MODEL_PATH='./Config'
    # TOKENIZER_PATH = './Config'
    #
    # # 准备模型，优化器和loss函数
    # model = BertClassifier(MODEL_PATH)
    # optimizer_name = 'Adam'
    # criterion_name = 'CrossEntropyLoss'
    # device=local_rank
    #
    # # 准备数据集中一个简单的例子
    # df = load_data("Data/data.jsonl")
    # df_train, df_val, df_test = split_data(df)
    # tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)
    # test_data = EmotionDataset(tokenizer, df_test)
    # text, label = test_data[0]
    # input_and_label = {
    #     'input':
    #         {
    #         'attention_mask': text['attention_mask'],
    #         'input_ids': text['input_ids']
    #         },
    #     'label':label
    # }
    # total_batchsize=1000



    #prepare for alexnet
    model = AlexNet(5)
    optimizer_name = 'Adam'
    criterion_name = 'CrossEntropyLoss'
    device = local_rank

    test_input = torch.FloatTensor(1, 3, 449, 449)
    test_label = torch.LongTensor([1])
    input_and_label = \
        {
             'input':test_input,
             'label':test_label
        }
    total_batchsize = 2048

    max_batchsize_list=Catch_MaxBatchsize(model=model,
                                          optimizer_name=optimizer_name,
                                          criterion_name=criterion_name,
                                          device=device,
                                          input_label=input_and_label,
                                          Total_Batchsize=total_batchsize,
                                          global_rank=global_rank,
                                          local_rank=local_rank,
                                          world_size=world_size
                                          )




    assert (len(max_batchsize_list) == len(initial_batch_list))
    final_execution_list=batchsize_scheduling(global_rank, local_rank, initial_batch_list, max_batchsize_list)
    print("The final execution is ",final_execution_list)

if __name__ == '__main__':
    main()