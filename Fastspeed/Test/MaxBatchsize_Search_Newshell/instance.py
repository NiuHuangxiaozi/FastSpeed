import sys
import os
import torch
import torch.nn as nn
import argparse
from argparse import Namespace
from torch.optim import Adam,SGD
from typing import List



class Test_max_batchsize:
    def __init__(self,model,optimizer_name:str,criterion_name:str,device):
        self.input=None
        self.label=None
        self.optimizer_name=optimizer_name
        self.criterion_name=criterion_name
        self.device=device
        self.model = model

    #IOC注入
    def setInputData(self,input):
        self.input = input
    def setLabel(self,label):
        self.label=label

    def _repeat_size(self,_size:torch.Size,multiplier)->List[int]:
        shape_list=list(_size)
        multiplier_list=[multiplier]
        for _ in range(len(shape_list)-1):
            multiplier_list.append(1)
        return multiplier_list

    def _construct_optim(self,model):
        if self.optimizer_name=='Adam':
            return Adam(model.parameters(),lr=1e-6)
        else:
            print("Error in Test_max_batchsize _construct_optim: wrong optim name!")
            sys.exit(1)
    def _construct_criterion(self):
        if self.criterion_name=='CrossEntropyLoss':
            return nn.CrossEntropyLoss()
        else:
            print("Error in Test_max_batchsize _construct_criterion: wrong loss function!")
            sys.exit(1)

    def is_oom(self,batchsize):
        try:
            #input
            if isinstance(self.input, dict):
                batchsized_input={}
                for key in self.input:
                    item=self.input[key]
                    batchsized_input[key]=item.clone().detach().repeat(self._repeat_size(item.shape,batchsize)).to(self.device)

            elif isinstance(self.input, torch.Tensor):
                    batchsized_input=self.input.clone().detach().repeat(self._repeat_size(self.input.shape,batchsize)).to(self.device)
            else:
                raise  TypeError("self.input must be torch.Tensor or a torch.Tensor dict!")

            #label
            if isinstance(self.label, torch.Tensor):
                 batchsized_label=self.label.clone().detach().repeat(self._repeat_size(self.label.shape,batchsize)).to(self.device)
            else:
                raise TypeError("self.label must be torch.Tensor type!")

        except TypeError as e:
            print(e)
            sys.exit(1)


        try:
            gpu_model = self.model.to(self.device)
            optimizer=self._construct_optim(gpu_model)
            criterion=self._construct_criterion()
            # 前向传播
            output = gpu_model(batchsized_input)
            # 计算损失
            batch_loss = criterion(output,batchsized_label)
            # 清除梯度
            optimizer.zero_grad()
            # 后向传播
            batch_loss.backward()
            # 梯度更新
            optimizer.step()

            #归还显存
            del gpu_model
            del optimizer
            del criterion
            del batchsized_input
            del batchsized_label
            del batch_loss
            torch.cuda.empty_cache()
            return False
        except RuntimeError as exception:
                if "out of memory" in str(exception):
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                        return True

    def search_max_batchsize(self,max_batchsize:int)->int:
        left, right = 1,max_batchsize
        while left <= right:
            mid = (right - left) // 2 + left
            flag=self.is_oom(mid)
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



########################################################################################################################
##do not delete the following comments

from  model import *





OPTIMIZER_NAME="Adam"
CRITERION_NAME="CrossEntropyLoss"
DEVICE=0
CEILING_BATCHSIZE=1000
TEMPSAMPLE_PATH= "Data/cifar_testsample.pt"

########################################################################################################################

def main():
    model=AlexNet(*[10])
    optimizer_name=OPTIMIZER_NAME
    criterion_name=CRITERION_NAME
    device=DEVICE
    test_platform=Test_max_batchsize(model,optimizer_name,criterion_name,device)

    # 准备数据
    input_and_label=torch.load(TEMPSAMPLE_PATH)
    one_input=input_and_label["input"]
    one_label=input_and_label["label"]

    test_platform.setInputData(one_input)
    test_platform.setLabel(one_label)
    max_batchsize = test_platform.search_max_batchsize(CEILING_BATCHSIZE)
    print(max_batchsize,end='')


if __name__=="__main__":
    main()