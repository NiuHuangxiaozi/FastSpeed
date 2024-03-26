import sys

import torch
import torch.nn as nn
from Data.load_data import load_data,split_data,EmotionDataset
from transformers import BertTokenizer
from model import BertClassifier,AlexNet
from torch.optim import Adam
from typing import List
import copy
import subprocess






def list2str(s:List[str])->str:
    result=""
    for item in s:
        result+=(item+'\n')
    return result

def repalce_macro(data:str,d:dict)->str:
    for initial_value in d:
        data=data.replace(initial_value,d[initial_value])
    return data






def search_max_batchsize(max_batchsize:int)->int:
    ########################################################################################################
    # 这里新创建一个进程执行检测
    device = 0
    file_path = "./template.py"
    target_file_path = "./instance.py"
    data = None
    dependency_list = ["from model import BertClassifier"]
    macros = {
        "#__MODEL_NAME__#": "BertClassifier",
        "#__MODEL_ARGS__#": "*['./Config']",
        "#__OPTIMIZER_NAME__#": "Adam",
        "#__CRITERION_NAME__#": "CrossEntropyLoss",
        "#__DEVICE__#": str(device),
        "#__CEILING_BATCHSIZE__#": str(max_batchsize),
        "#__TEMPSAMPLE_PATH__#": "./Data/testsample.pt"
    }
    cmd = ["python", target_file_path]
    ########################################################################################################
    left, right = 1,max_batchsize
    while left <= right:
        mid = (right - left) // 2 + left
        ########################################################################################################
        ########################################################################################################
        macros["#__CEILING_BATCHSIZE__#"]=str(mid)
        with (open(file_path, 'r', encoding='utf-8') as f):
            data = f.read()
            data = data.replace("#__Dependency__#", list2str(dependency_list))
            data = repalce_macro(data, macros)

        with open(target_file_path, 'w', encoding='utf-8') as f:
            f.write(data)

        result = subprocess.run(cmd, stdout=subprocess.PIPE)
        flag = int(result.stdout.decode('utf-8'))
        print("The omm detection is ",flag)
        ######################################################################################################
        ########################################################################################################
        if left ==right:
            if flag:
                return mid-1
            else:
                return mid
        elif left+1 ==right:
            if flag:
                return mid-1
            else:
                left=mid+1
        else:
            if flag:
                right=mid-1
            else:
                left=mid+1
    assert(False)
    return 0;



def main():
    answer=search_max_batchsize(100)
    print("The limit batchsize is ",answer)
if __name__=="__main__":
    main()