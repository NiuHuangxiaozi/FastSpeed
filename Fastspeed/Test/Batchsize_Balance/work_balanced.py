import sys

import torch
import torch.distributed as dist
import heapq
import copy
import os
from torch.distributed import init_process_group,destroy_process_group
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

def batchsize_scheduling(global_rank,local_rank,initial_batch_list,max_batchsize_list):
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
    Task(local_rank,global_rank)
    print("RANK :", global_rank, "All Finished")
    Distributed_destroy()


def Task(local_rank,global_rank):
    initial_batch_list=[5,15,20,25,35,45,50,55,60,65]
    max_batchsize_list=[5,16,21,26,36,46,51,56,61,55]
    batchsize_scheduling(global_rank,local_rank, initial_batch_list, max_batchsize_list)


if __name__ == '__main__':
    main()