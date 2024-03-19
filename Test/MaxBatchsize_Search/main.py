import sys

import torch
import torch.nn as nn
from Data.load_data import load_data,split_data,EmotionDataset
from transformers import BertTokenizer
from model import BertClassifier,AlexNet
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
                    batchsized_input[key]=copy.deepcopy(item).repeat(self._repeat_size(item.shape,batchsize)).to(self.device)

                    print(batchsized_input[key].shape)
            elif isinstance(self.input, torch.Tensor):
                    batchsized_input=copy.deepcopy(self.input).repeat(self._repeat_size(self.input.shape,batchsize)).to(self.device)
                    print(batchsized_input.shape)
            else:
                raise  TypeError("self.input must be torch.Tensor or a torch.Tensor dict!")

            #label
            if isinstance(self.label, torch.Tensor):
                 batchsized_label=copy.deepcopy(self.label).repeat(self._repeat_size(self.label.shape,batchsize)).to(self.device)
            else:
                raise TypeError("self.label must be torch.Tensor type!")

            print(batchsized_label.shape)
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
                print(str(exception))
                if "out of memory" in str(exception):
                        print('In batchsize:'+str(batchsize)+' WARNING: out of memory')
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                        return True


    def search_max_batchsize(self,max_batchsize:int)->int:
        left, right = 1,max_batchsize
        while left <= right:
            mid = (right - left) // 2 + left
            flag=self.is_oom(mid)
            print(flag)
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




def test_bert_base_uncased():
    print("Begin to test max batchsize.")
    print("Begin to load data.")

    #准备训练平台的一些参数
    MODEL_PATH = './Config/'
    model = BertClassifier(MODEL_PATH)
    optimizer_name ='Adam'
    criterion_name = 'CrossEntropyLoss'
    device =1

    test_platform = Test_max_batchsize(model, optimizer_name, criterion_name, device)


    #准备数据
    TOKENIZER_PATH='./Config/'
    df = load_data("Data/data.jsonl")
    df_train, df_val, df_test = split_data(df)
    tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)
    test_data = EmotionDataset(tokenizer, df_test)
    text,label=test_data[0]
    print(type(text['attention_mask']))
    print(type(text['input_ids']))
    print(type(label))
    input = \
    {
        'attention_mask':text['attention_mask'],
        'input_ids': text['input_ids']
    }
    test_platform.setInputData(input)
    test_platform.setLabel(label)

    max_batchsize = test_platform.search_max_batchsize(5000)
    print("The max batchsize is ", max_batchsize)



def test_alexnet():
    model=AlexNet(5)
    optimizer_name = 'Adam'
    criterion_name = 'CrossEntropyLoss'
    device = 1
    test_platform = Test_max_batchsize(model, optimizer_name, criterion_name, device)

    test_input=torch.FloatTensor(1,3,449,449)
    test_label=torch.LongTensor([1])
    test_platform.setInputData(test_input)
    test_platform.setLabel(test_label)

    max_batchsize = test_platform.search_max_batchsize(2048)
    print("The max batchsize is ", max_batchsize)
def main():
    choice='bert'
    if choice=='bert':
        test_bert_base_uncased()
    elif choice=='Alexnet':
        test_alexnet()
if __name__=="__main__":
    main()