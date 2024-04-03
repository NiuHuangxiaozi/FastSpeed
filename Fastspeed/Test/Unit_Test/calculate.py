
#third party
import torch
import time
import sys
sys.path.append('C:\\D\\Lesson\\Final_Design\\')
import torch.nn as nn
import torch.optim as optim
from thop import profile,clever_format


########################################################################################################################
# the codes are going to be tested
def calculate_time(model, shape: tuple):
    # create a temporary tensor
    temp_tensor = torch.normal(0, 1, size=shape).to(next(model.parameters()).device)
    torch.cuda.synchronize()
    start = time.time()
    model.train()
    result = model(temp_tensor)
    torch.cuda.synchronize()
    end = time.time()
    time_cost = end - start
    return time_cost


def calculate_memmory(model,
                      criterion,
                      optimizer_choice: str,
                      shape: tuple,
                      device):
    torch.cuda.empty_cache()  # 将所有缓冲区的显存分配去除
    torch.cuda.synchronize()  # GPU等待所有的GPU事务完成
    before_memory = torch.cuda.memory_allocated(device)
    print("initial_torch_memory", before_memory)

    temp_tensor = torch.normal(0, 1, size=shape).to(device)  # 创建一个简单的张量
    print("aftersample_torch_memory", torch.cuda.memory_allocated(device))
    model.train()
    model = model.to(device)
    print("aftermodel_torch_memory", torch.cuda.memory_allocated(device))

    try:
        if optimizer_choice == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=0.001, )
        else:
            raise ValueError
    except ValueError:
        print("In calculation.py calculate_memmory function optimizer_choice is wrong!")
    optimizer.zero_grad()
    print("beforetrain_torch_memory", torch.cuda.memory_allocated(device))
    result = model(temp_tensor)
    print("aftertrain_torch_memory", torch.cuda.memory_allocated(device))
    torch.cuda.empty_cache()
    loss = criterion(result, result - 1e-4)
    print("aftercriterion_torch_memory", torch.cuda.memory_allocated(device))
    loss.backward()
    print("afterbackward_torch_memory", torch.cuda.memory_allocated(device))
    optimizer.step()
    print("afterstep_torch_memory", torch.cuda.memory_allocated(device))
    torch.cuda.synchronize()
    after_memory = torch.cuda.max_memory_allocated(device)
    print("final_torch_memory", after_memory)
    return after_memory - before_memory


def cnnCalculateParam(model,shape):
    dummy_input = torch.normal(0, 1, size=shape)
    flops, params = profile(model, inputs=(dummy_input,))
    return flops*2 , params

########################################################################################################################





# test 1
def test1():
    # 模型放到第零个GPU上
    model = AlexNet(num_classes=4).to(0)
    shape = (28, 3, 449, 449)
    time = calculate_time(model, shape)
    print("The time is :", time)





# test 2
class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.net1 = torch.nn.Linear(1, 1,bias=False)
        self.activation_sum = 0
        self.activation_size = 0

    def forward(self, x):
        x = self.net1(x)
        self.activation_sum += x.nelement()
        self.activation_size += (x.nelement() * x.element_size())
        return x
def test2():
    # 测试最大显存占用
    shape = (1, 1)
    model = MyModel()
    processed_memory = calculate_memmory(model, nn.CrossEntropyLoss(), 'Adam', shape, 0)
    print(model.activation_sum)
    print(model.activation_size)



#test3:
'''
经过测试，profile库要是统计参数量的话必须写进class MyModel(torch.nn.Module):的类里面，
如果直接model=nn.Linear(1,1,bias=False)就统计不出参数量
'''

#原文链接：https: // blog.csdn.net / qq_41979513 / article / details / 102369396
def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

class testLinear(torch.nn.Module):
    def __init__(self):
        super(testLinear,self).__init__()
        self.linear=nn.Linear(15,10,bias=True)
        self.linear1=nn.Linear(10,4,bias=True)
    def forward(self,x):
        x=self.linear(x)
        x=self.linear1(x)
        return x
class testConv(torch.nn.Module):
    def __init__(self):
        super(testConv,self).__init__()
        self.conv=nn.Conv2d(in_channels=3,out_channels=64,padding=0,stride=1,kernel_size=3)
    def forward(self,x):
        return self.conv(x)
class testSelfAttention(torch.nn.Module):
    def __init__(self):
        super(testSelfAttention, self).__init__()
        self.transformer=nn.Transformer(d_model=512, batch_first=True)
    def forward(self,x):
        return self.transformer(x,x)
        
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

'''
这个有许多的计算计算复杂度的库，包括：pytorchstat，ptflops，thop,在github上的star比较多，这里选用thop.
这里发现，使用上面的ptflops和thop统计transformer时不一样，可能是他们计算的方式不同吧。
'''
def test3():
    #统计线性层
    model1=testLinear()
    print("profile计算：线性层的计算量和参数量为：",cnnCalculateParam(model1,(1,15)))
    print("手动计算：线性层的参数量为：",get_parameter_number(model1))

    #统计卷积层
    model2 = testConv()
    print("profile计算：卷积层的计算量和参数量为：", cnnCalculateParam(model2, (2,3,32,32)))
    flops,params=cnnCalculateParam(model2, (2,3,32,32))
    flops, params = clever_format([flops, params], '%.3f')
    print(f"运算量：{flops}, 参数量：{params}")
    
    print("手动计算：卷积层的参数量为：", get_parameter_number(model2))

    #统计transformer
    model3 = testSelfAttention()
    print("profile计算：transformer的计算量和参数量为：", cnnCalculateParam(model3, (2, 3, 512)))
    print("手动计算：transformer的参数量为：", get_parameter_number(model3))


if __name__ == '__main__':
    test3()