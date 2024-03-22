import sys
import torch
import heapq
import copy
from typing import List
import torch.distributed as dist
#  my lib
from help.debug import MyException

GREEN = '\033[92m'
END_GREEN = '\033[0m'

__all__ = ["batchsize_scheduling", ]


########################################################################################################################
# 关于打印的函数

def _print_red_symbol(s: str) -> None:
    print("\033[0;31;40m", s, "\033[0m")


def _print_current_batchconfig(initial_list, max_batch_list):
    initial_config = {
        'GPU ' + str(index) + " : " + str(initial_list[index]) for index in range(len(initial_list))
    }
    maxbatch_config = {
        'GPU ' + str(index) + " : " + str(max_batch_list[index]) for index in range(len(max_batch_list))
    }
    _print_red_symbol("The best batchsize config is %s" % (str(initial_config)))
    _print_red_symbol("The max batchsize limit is %s" % (str(maxbatch_config)))


# 数据移动函数
def _shift_load(free_devices,
                omm_devices,
                final_batch_list,
                demand_batch,
                peak_device,
                left_batch,
                valley_device):
    if demand_batch < left_batch:
        heapq.heappush(free_devices, (-(left_batch - demand_batch), valley_device))
        final_batch_list[valley_device] += demand_batch
        final_batch_list[peak_device] -= demand_batch
    elif demand_batch == left_batch:
        final_batch_list[valley_device] += demand_batch
        final_batch_list[peak_device] -= demand_batch
    else:
        heapq.heappush(omm_devices, (-(demand_batch - left_batch), peak_device))
        final_batch_list[valley_device] += left_batch
        final_batch_list[peak_device] -= left_batch

    return free_devices, omm_devices, final_batch_list


# 数据负载均衡
def batchsize_scheduling(global_rank, local_rank, initial_batch_list, max_batchsize_list) -> List[int]:
    if global_rank == 0:
        assert len(initial_batch_list) == len(max_batchsize_list)
        final_batch_list = copy.deepcopy(initial_batch_list)
        try:
            free_devices = []
            omm_devices = []
            for index in range(len(final_batch_list)):
                if final_batch_list[index] > max_batchsize_list[index]:
                    heapq.heappush(omm_devices, (-(final_batch_list[index] - max_batchsize_list[index]), index))
                elif final_batch_list[index] < max_batchsize_list[index]:
                    heapq.heappush(free_devices, (-(max_batchsize_list[index] - final_batch_list[index]), index))

            while (len(free_devices) != 0) and (len(omm_devices) != 0):
                demand_batch, peak_device = heapq.heappop(omm_devices)
                left_batch, valley_device = heapq.heappop(free_devices)
                free_devices, omm_devices, final_batch_list = _shift_load(free_devices, omm_devices,
                                                                          final_batch_list,
                                                                          -demand_batch, peak_device,
                                                                          -left_batch, valley_device)
            if len(omm_devices) != 0 and global_rank == 0:
                _print_current_batchconfig(initial_batch_list, max_batchsize_list)
                raise MyException(0, "Total batchsize %d can't be train on the system." % (sum(initial_batch_list)))
            final_batch_list = torch.IntTensor(final_batch_list).to(local_rank)
            dist.broadcast(tensor=final_batch_list, src=0)
            print(GREEN, 'After broadcast, Rank ', global_rank, ' has data ', str(final_batch_list), END_GREEN)
        except MyException as e:
            _print_red_symbol(e)
            sys.exit(1)
    else:
        final_batch_list = torch.IntTensor(len(initial_batch_list)).to(local_rank)
        dist.broadcast(tensor=final_batch_list, src=0)
        if global_rank == 1:
            print(GREEN, 'After broadcast, Rank ', global_rank, ' has data ', str(final_batch_list), END_GREEN)

    return final_batch_list