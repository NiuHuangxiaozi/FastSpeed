import sys
import torch
import torch.nn as nn
from torch.optim import Adam
from typing import List
import copy

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
        if self.criterion_name=='CrossEntropy':
            return nn.CrossEntropyLoss()
        else:
            print("Error in Test_max_batchsize _construct_criterion: wrong loss function!")
            sys.exit(1)

    def is_oom(self,batchsize)->int:
        try:
            #input
            if isinstance(self.input, dict):
                batchsized_input={}
                for key in self.input:
                    item=self.input[key]
                    batchsized_input[key]=copy.deepcopy(item).repeat(self._repeat_size(item.shape,batchsize)).to(self.device)
            elif isinstance(self.input, torch.Tensor):
                    batchsized_input=copy.deepcopy(self.input).repeat(self._repeat_size(self.input.shape,batchsize)).to(self.device)
            else:
                raise  TypeError("self.input must be torch.Tensor or a torch.Tensor dict!")

            #label
            if isinstance(self.label, torch.Tensor):
                 batchsized_label=copy.deepcopy(self.label).repeat(self._repeat_size(self.label.shape,batchsize)).to(self.device)
            else:
                raise TypeError("self.label must be torch.Tensor type!")

        except TypeError as e:
            print(e)
            sys.exit(1)


        try:
            gpu_model = self.model.to(self.device)
            optimizer=self._construct_optim(gpu_model)
            criterion=self._construct_criterion()
            gpu_model.train()
            for epoch in range(2):
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
            return 0
        except RuntimeError as exception:
                if "out of memory" in str(exception):
                        return 1
                elif "GET was unable to find an engine to execute this computation" in str(exception):
                        return 1
                else:
                        print(exception)
                        sys.exit(1)


import os
sys.path.append(os.getcwd())
#__Dependency__#




OPTIMIZER_NAME="#__OPTIMIZER_NAME__#"
CRITERION_NAME="#__CRITERION_NAME__#"
DEVICE=#__DEVICE__#
CEILING_BATCHSIZE=#__CEILING_BATCHSIZE__#
TEMPSAMPLE_PATH="#__TEMPSAMPLE_PATH__#"




def main():
    # 准备训练平台的一些参数
    model =#__MODEL_NAME__#(#__MODEL_ARGS__#)
    optimizer_name = OPTIMIZER_NAME
    criterion_name = CRITERION_NAME
    device = DEVICE


    test_platform = Test_max_batchsize(model, optimizer_name, criterion_name, device)

    # 准备数据
    input_and_label = torch.load(TEMPSAMPLE_PATH)
    one_input = input_and_label["input"]
    one_label = input_and_label["label"]

    test_platform.setInputData(one_input)
    test_platform.setLabel(one_label)

    flag = test_platform.is_oom(CEILING_BATCHSIZE)
    print(flag)

if __name__=='__main__':
    main()