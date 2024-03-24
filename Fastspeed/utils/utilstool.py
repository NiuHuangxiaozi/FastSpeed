import argparse
import os
import copy
import json
import sys

import torch
from Fastspeed.help.debug import MyException
from typing import List
'''
这里的代码来源于：https://blog.csdn.net/kiong_/article/details/135492019
'''
class Params():
    def __init__(self,json_path):
            with open(json_path) as f:
                params = json.load(f)
                self.__dict__.update(params)

def to_cpu(obj):
    if isinstance(obj, dict):
        for key in obj:
            obj[key]=obj[key].cpu()
    elif isinstance(obj, torch.Tensor):
            obj=obj.cpu()
    else:
        assert(False)
    return obj


def to_device(obj,device:int):
    if isinstance(obj, dict):
        for key in obj:
            obj[key]=obj[key].to(device)
    elif isinstance(obj, torch.Tensor):
            obj=obj.to(device)
    else:
        assert(False)
    return obj



# multiply on batchsize dim
def repeat_obj(input,multiplyer):
    # input
    try:
        if isinstance(input, dict):
            result = {}
            for key in input:
                item = input[key]
                result[key] = item.clone().detach().repeat(_repeat_batch(item.shape, multiplyer))
        elif isinstance(input, torch.Tensor):
            result = input.clone().detach().repeat(_repeat_batch(input.shape, multiplyer))
        else:
            raise MyException("Error[in utilstool.py,repeat_obj]: Input must be torch.Tensor or a torch.Tensor dict!")
        return result

    except MyException as e:
        print(e)
        sys.exit(1)



#utilstool.py内部使用的函数
########################################################################################################################
def _repeat_batch(_size: torch.Size, multiplier) -> List[int]:
        shape_list = list(_size)
        multiplier_list = [multiplier]
        for _ in range(len(shape_list) - 1):
            multiplier_list.append(1)
        return multiplier_list



