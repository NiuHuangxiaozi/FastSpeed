
# official lib
import sys
import torch
import torch.nn as nn
import time
import torch.optim as optim
from thop import profile
from typing import List


# my lib
from Fastspeed.utils.utilstool import to_cpu,to_device,repeat_obj



__all__ = ["calculate_time", "calculate_memmory","calculate_param"]
def Get_LossFunction(name:str):
    if name=='CrossEntropy':
        return nn.CrossEntropyLoss()


#try to calculate the time
def calculate_time(model,criterion_name:str,device,input,label,test_batchsize:int):
    try:

        #create a temporary tensor
        temp_tensor=repeat_obj(input,test_batchsize)
        temp_label=repeat_obj(label,test_batchsize)
        temp_tensor=to_device(temp_tensor,device)
        temp_label =to_device(temp_label,device)

        #model
        model = model.to(device)

        #loss函数
        criterion=Get_LossFunction(criterion_name)

        torch.cuda.synchronize()
        start = time.time()
        result=model(temp_tensor)
        loss = criterion(result, result - 1e-4)
        loss.backward()
        torch.cuda.synchronize()
        end = time.time()
        time_cost=end-start


        #将模型卸载到cpu上，避免占用显存
        model=model.cpu()
        temp_tensor=to_cpu(temp_tensor)
        temp_label=to_cpu(temp_label)
        del temp_tensor
        del temp_label
        return time_cost
    except RuntimeError as exception:
        if "out of memory" in str(exception):
            print('Error[in calculation.py calculate_time]: Test time process is out of memory!')
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            sys.exit(1)



'''
这个计算模型占用的实际显存的函数在理论方面还有有完全验证成功，
还需要进一步实验
'''
def calculate_memmory(model,
                      criterion,
                      optimizer_choice: str,
                      shape: tuple,
                      device):
    torch.cuda.empty_cache()  # 将所有缓冲区的显存分配去除
    torch.cuda.synchronize()  # GPU等待所有的GPU事务完成
    before_memory = torch.cuda.memory_allocated(device)

    temp_tensor = torch.normal(0, 1, size=shape).to(device)  # 创建一个简单的张量
    model.train()
    model = model.to(device)
    try:
        if optimizer_choice == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=0.001, )
        else:
            raise ValueError
    except ValueError:
        print("In calculation.py calculate_memmory function optimizer_choice is wrong!")
    optimizer.zero_grad()
    result = model(temp_tensor)
    torch.cuda.empty_cache()
    loss = criterion(result, result - 1e-4)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    after_memory = torch.cuda.max_memory_allocated(device)
    return after_memory - before_memory



def cnnCalculateParam(model,shape):
    dummy_input = torch.normal(0, 1, size=shape)
    flops, params = profile(model, inputs=(dummy_input,))
    return flops*2 , params





'''
try:
    output = model(input)
except RuntimeError as exception:
    if "out of memory" in str(exception):
        print("WARNING: out of memory")
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
'''





