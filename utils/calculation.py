import torch
import torch.nn as nn
import time
import torch.optim as optim
from thop import profile
__all__ = ["calculate_time", "calculate_memmory","calculate_param"]


def Get_LossFunction(name:str):
    if name=='CrossEntropy':
        return nn.CrossEntropyLoss()

#try to calculate the time
def calculate_time(model,criterion_name:str,device,shape:tuple):

    #create a temporary tensor
    temp_tensor = torch.normal(0, 1, size=shape).to(device)

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
    return time_cost





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

def calculate_param(model,shape):
    dummy_input = torch.normal(0, 1, size=shape)
    flops, params = profile(model, inputs=(dummy_input,))
    return flops,params


'''
try:
    output = model(input)
except RuntimeError as exception:
    if "out of memory" in str(exception):
        print("WARNING: out of memory")
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
'''





