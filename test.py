import torch
from torch.utils.checkpoint import checkpoint


class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.net1 = torch.nn.Linear(3, 300)
        self.net2 = torch.nn.Linear(300, 300)
        self.net3 = torch.nn.Linear(300, 400)
        self.net4 = torch.nn.Linear(400, 300)
        self.net5 = torch.nn.Linear(300, 100)
        self.activation_sum = 0
        self.activation_size = 0

    def forward(self, x):
        x = self.net1(x)
        self.activation_sum += x.nelement()
        self.activation_size += (x.nelement() * x.element_size())
        x = self.net2(x)
        self.activation_sum += x.nelement()
        self.activation_size += (x.nelement() * x.element_size())
        x = self.net3(x)
        self.activation_sum += x.nelement()
        self.activation_size += (x.nelement() * x.element_size())
        x = self.net4(x)
        self.activation_sum += x.nelement()
        self.activation_size += (x.nelement() * x.element_size())
        x = self.net5(x)
        self.activation_sum += x.nelement()
        self.activation_size += (x.nelement() * x.element_size())
        return x


import torch
from torch.utils.checkpoint import checkpoint


class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.net1 = torch.nn.Linear(3, 300)
        self.net2 = torch.nn.Linear(300, 300)
        self.net3 = torch.nn.Linear(300, 400)
        self.net4 = torch.nn.Linear(400, 300)
        self.net5 = torch.nn.Linear(300, 100)
        self.activation_sum = 0
        self.activation_size = 0

    def forward(self, x):
        x = self.net1(x)
        self.activation_sum += x.nelement()
        self.activation_size += (x.nelement() * x.element_size())
        x = self.net2(x)
        self.activation_sum += x.nelement()
        self.activation_size += (x.nelement() * x.element_size())
        x = self.net3(x)
        self.activation_sum += x.nelement()
        self.activation_size += (x.nelement() * x.element_size())
        x = self.net4(x)
        self.activation_sum += x.nelement()
        self.activation_size += (x.nelement() * x.element_size())
        x = self.net5(x)
        self.activation_sum += x.nelement()
        self.activation_size += (x.nelement() * x.element_size())
        return x


def modelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size)
    return all_size


device = torch.device("cuda:0")

input = torch.randn(10, 3).to(device)
label = torch.randn(10, 100).to(device)

torch.cuda.empty_cache()
before = torch.cuda.memory_allocated()
model = MyModel().to("cuda:0")
after = torch.cuda.memory_allocated()
print("建立模型后显存变大{}".format(after - before))

print("模型大小为{}".format(modelSize(model)))

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
model.train()
optimizer.zero_grad()

before = torch.cuda.memory_allocated()
print("模型前向传播前使用显存为{}".format(before))

output = model(input)  # 前向传播

after = torch.cuda.memory_allocated()
print("模型前向传播后使用显存为{}，差值（中间激活）为{}".format(after, after - before))

loss = loss_fn(output, label)
torch.autograd.backward(loss)
optimizer.step()

print(model.activation_sum)
print(model.activation_size)
