# 自己写的其他的库
from Fastspeed.utils.sampler import Distributed_Elastic_Sampler
from Fastspeed.help.debug import global_rank_print
from Fastspeed.utils.calculation import calculate_time
from Fastspeed.utils.resource_detection import Test_MaxBatchsize
from Fastspeed.utils.utilstool import Params, list2str, repalce_macro
from argparse import Namespace
from Fastspeed.utils.scheduling import batchsize_scheduling
from Fastspeed.utils.utilstool import to_device
# exception
from Fastspeed.help.debug import MyException

# 官方的库
import os
import contextlib
import copy
import sys
import torch
import time
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import List
import torch.distributed as dist
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import subprocess

'''
这个文件的作用是实现数据集的合理划分以及模型流水线并行的合理划分
'''
__all__ = ["FastSpeed", ]


class FastSpeed:

    def __init__(self, param_config: Params,
                 args: Namespace,
                 test_input,
                 test_label) -> None:
        self.grad_portion: List[float] = None  # 对于数据并行不平衡的batch的normalize
        # 用来计算模型的运行时间并做一些分析
        self.start_time = 0.0
        self.end_time = 0.0

        # 统一每一个epoch的iter数，防止有些节点训练完毕，有些没有训练完毕
        self.max_epoch_iter: int = 0
        self.epoch_iters: List[int] = []
        self.dummy_input = None  # 以下两个参数，用作在iter不够的时候顶上去
        self.dummy_label = None

        # 保存参数
        self.config: Params = param_config
        self.dist_args: Namespace = args
        self.test_input = test_input
        self.test_label = test_label

        # 保存用于临时样例
        InputLabel = {
            "input": self.test_input,
            "label": self.test_label
        }
        PATH = "./Fastspeed/Temp/" + self.config.strategy["model_name"] + "_testsample.pt"
        torch.save(InputLabel, PATH)

    '''
    model_wrap:构建我们需要训练的模型
    '''

    def model_wrap(self, model):
        m_type: str = self.config.strategy['model_type']
        device = self.dist_args.local_rank
        try:
            if m_type == 'ddp':
                self.wrapped_model = DDP(model.to(device))
            elif m_type == 'pipe':
                pass
            elif m_type == 'hybrid':
                pass
            else:
                raise MyException \
                    ("Error:[in FastSpeed.model_wrap] Current model parallel type only contain ddp, pipe and hybrid(ddp,pipe)")
        except MyException as e:
            print(e)
            sys.exit(1)
        return self.wrapped_model

    # 获取模型最好的运行数据分配
    def _autobalance(self, model) -> List[int]:
        m_type: str = self.config.strategy['model_type']
        batchsize_list = None
        try:
            if m_type == 'ddp':
                dummy_model = copy.deepcopy(model)  # 独立拷贝一份模型用作测试防止修改原来的模型
                if self.config.strategy["unbalanced_strategy"] == "time":
                    # 获得运行时间
                    dummy_time = calculate_time(dummy_model,
                                                self.config.train["criterion"],
                                                self.dist_args.local_rank,
                                                self.test_input,
                                                self.test_label,
                                                self.config.strategy["test_batchsize_bytime"]
                                                )

                    dummy_time_tensor = torch.tensor(dummy_time).to(self.dist_args.local_rank)

                    time_list = [torch.zeros(1, ).to(self.dist_args.local_rank) for i in
                                 range(0, self.dist_args.world_size)]

                    print("time_list", time_list)
                    print("dummy_time_tensor", dummy_time_tensor)

                    dist.all_gather(time_list, dummy_time_tensor)
                    print("time_list", time_list)

                    time_list = [(1 / obj.item()) * 100 for obj in time_list]
                    rounded_time_array = np.trunc(time_list) / 100

                    print("rounded_time_array", rounded_time_array)
                    portion = rounded_time_array / np.sum(rounded_time_array)
                    print("portion", portion)

                    batchsize_list = (self.config.train["batch_size"] * portion).astype(int)
                    batchsize_list[-1] = self.config.train["batch_size"] - np.sum(batchsize_list[:-1])
                    return batchsize_list.tolist()
                else:
                    raise MyException(2, "The auto balanced strategy is only support for time and complexity now.")
            elif m_type == 'pipe':
                pass
            else:
                raise MyException(3, "The parallel type only support for ddp ,pipe or hybrid")
        except MyException as e:
            print(e)
            sys.exit(1)

    # 获取最大能承受的batchsize,0号节点使用gather函数获取所有节点的最大batchsize组成一个list
    def Catch_MaxBatchsize(self, model) -> List[int]:
        # print("Begin to test max batchsize.")
        # # 准备后面的参数
        # optimizer_name =self.config.train["optimizer"]
        # criterion_name =self.config.train["criterion"]
        # device =self.dist_args.local_rank
        # Total_Batchsize =self.config.train["batch_size"]
        #
        # test_platform = Test_MaxBatchsize(model, optimizer_name, criterion_name, device)
        #
        # test_platform.setInputData(self.test_input)
        # test_platform.setLabel(self.test_label)
        #
        # max_batchsize = test_platform.search_max_batchsize(Total_Batchsize)
        # print("The max batchsize is ", max_batchsize)
        # # gather_list用于收集所有的max_batchsize
        #
        # MaxBatchsize_Tensor = torch.IntTensor([max_batchsize]).to(device)
        # if self.dist_args.global_rank == 0:
        #     gather_list = [torch.IntTensor([1]).to(device) for _ in range(self.dist_args.world_size)]
        #     dist.gather(MaxBatchsize_Tensor, dst=0, gather_list=gather_list)
        #
        #     print("All the max batchsize is ", gather_list)
        #     max_batchsize_list = [tensor_val.item() for tensor_val in gather_list]
        # else:
        #     max_batchsize_list = [0 for _ in range(self.dist_args.world_size)]
        #     dist.gather(MaxBatchsize_Tensor, dst=0)
        ################################################################################################################
        max_batchsize=-1

        device = self.dist_args.local_rank
        file_path = "./Fastspeed/Temp/template.py"
        target_file_path = "./Fastspeed/Temp/instance.py"
        data = None
        dependency_list = self.config.strategy["runing_dependency"]
        macros = {
            "#__MODEL_NAME__#": self.config.strategy["model_name"],
            "#__MODEL_ARGS__#": "*" + str(self.config.strategy["model_args"]),
            "#__OPTIMIZER_NAME__#": self.config.train["optimizer"],
            "#__CRITERION_NAME__#": self.config.train["criterion"],
            "#__DEVICE__#": str(device),
            "#__CEILING_BATCHSIZE__#": str(self.config.train["batch_size"]),
            "#__TEMPSAMPLE_PATH__#": "./Fastspeed/Temp/" + self.config.strategy["model_name"] + "_testsample.pt"
        }
        cmd = ["python", target_file_path]
        left, right = 1, self.config.train["batch_size"]
        while left <= right:
            mid = (right - left) // 2 + left
            #-----------------------------------------------------------------------------------------------------------
            macros["#__CEILING_BATCHSIZE__#"] = str(mid)
            with (open(file_path, 'r', encoding='utf-8') as f):
                data = f.read()
                data = data.replace("#__Dependency__#", list2str(dependency_list))
                data = repalce_macro(data, macros)

            with open(target_file_path, 'w', encoding='utf-8') as f:
                f.write(data)

            #等待所有的cuda任务全部执行完
            torch.cuda.synchronize()

            result = subprocess.run(cmd, stdout=subprocess.PIPE)
            flag = int(result.stdout.decode('utf-8'))
            print("The omm detection is ", flag)

            #-----------------------------------------------------------------------------------------------------------
            if left == right:
                if flag:
                    max_batchsize=mid - 1
                else:
                    max_batchsize = mid
                break
            elif left + 1 == right:
                if flag:
                    max_batchsize = mid - 1
                    break
                else:
                    left = mid + 1
            else:
                if flag:
                    right = mid - 1
                else:
                    left = mid + 1

        assert(max_batchsize!=-1)
        print("The max batchsize is ", max_batchsize)
        # --------------------------------------------------------------------------------------------------------------

        # gather_list用于收集所有的max_batchsize
        MaxBatchsize_Tensor = torch.IntTensor([max_batchsize]).to(device)
        if self.dist_args.global_rank == 0:
            gather_list = [torch.IntTensor([1]).to(device) for _ in range(self.dist_args.world_size)]
            dist.gather(MaxBatchsize_Tensor, dst=0, gather_list=gather_list)
            max_batchsize_list = [tensor_val.item() for tensor_val in gather_list]
        else:
            max_batchsize_list = [0 for _ in range(self.dist_args.world_size)]
            dist.gather(MaxBatchsize_Tensor, dst=0)
        print("The global rank is ", self.dist_args.global_rank, " And the max_batchsize_list is ", max_batchsize_list)
        return max_batchsize_list

    # 对数据进行异构的划分
    def unbalanced_datasplit(self, dataset, model) -> torch.utils.data.DataLoader:
        partition_type = self.config.strategy["partition_type"]
        TotalDataNumber = self.config.data["total_datasize"]
        try:
            if self.config.data["total_datasize"] != len(dataset):
                raise MyException \
                    ("Error[Fastspeed,unbalanced_datasplit]:The  length of dataset dismatch with json file config.")
        except MyException as e:
            print(e)
            sys.exit(1)
        try:
            if partition_type == 'manual' or partition_type == 'autobalanced':
                # 0
                ########################################################################################################
                # 这一段选择我们的partition_list和batchsize_list是从哪里来，是人工还是系统自己计算
                if partition_type == 'autobalanced':
                    print("This is autobalanced partition mode!")

                    # 获取最理想的数据分配
                    batchsize_limit_list = self.Catch_MaxBatchsize(model=model)
                    torch.cuda.synchronize()
                    initial_batchsize_list = self._autobalance(model=model)

                    batchsize_list = batchsize_scheduling(self.dist_args.global_rank,
                                                          self.dist_args.local_rank,
                                                          initial_batchsize_list,
                                                          batchsize_limit_list)
                    # temp_portion 是 numpy对象
                    batchsize_list = np.array(batchsize_list.cpu())
                    temp_portion = batchsize_list / np.sum(batchsize_list)
                    partition_list = (TotalDataNumber * temp_portion).astype(int)
                    partition_list[-1] = TotalDataNumber - np.sum(partition_list[:-1])

                    partition_list = partition_list.astype(int).tolist()
                    batchsize_list = batchsize_list.astype(int).tolist()
                else:
                    print("This is manual partition mode!")
                    partition_list, batchsize_list = self.config.strategy["manual_partition_list"] \
                        , self.config.strategy["manual_batchsize_list"]
                ########################################################################################################

                # 1
                ########################################################################################################
                # 统一每一个epoch的iter数，防止有些节点训练完毕，有些没有训练完毕
                self.epoch_iters = np.ceil(np.divide(partition_list, batchsize_list)).astype(int)
                self.max_epoch_iter = max(self.epoch_iters)
                ########################################################################################################

                # 2
                ########################################################################################################
                global_rank_print(0, "epoch_iters is " + str(list(self.epoch_iters)))
                global_rank_print(0, "max_epoch_iter is " + str(self.max_epoch_iter))
                # for debug
                global_rank_print(0, "The partition_list is : " + str(partition_list))
                global_rank_print(0, "The batchsize_list is : " + str(batchsize_list))
                # for debug
                if partition_list is None or batchsize_list is None:
                    raise MyException(0, "partition_list or batchsize_list can't be None in the manual mode!")
                ########################################################################################################

                # 3
                ########################################################################################################
                # 这里在计算完batchsize_list之后我们需要计算一下各个梯度的权重比例
                # 然后调用Distributed_Elastic_Sampler和相应的dataloader进行数据的划分
                self.grad_portion = batchsize_list / np.sum(batchsize_list)

                sampler_dict = \
                    {
                        'method': "uneven",
                        'partition_list': partition_list
                    }

                sampler = Distributed_Elastic_Sampler(dataset=dataset, partition_strategy=sampler_dict)
                train_loader = DataLoader(dataset=dataset,
                                          batch_size=batchsize_list[self.dist_args.global_rank],
                                          shuffle=False,  # 这个值必须设置为false，否则会导致多个节点可能都抽到同一个样例的结果
                                          sampler=sampler,
                                          pin_memory=self.config.data["train_loader_pin_memory"],
                                          num_workers=self.config.data["train_loader_num_workers"]
                                          )
                return train_loader
            else:
                raise MyException(1,
                                  "Your choice of partition_method(" + partition_type + ") is wrong [neither manual or autobalanced!")
                ########################################################################################################
        except MyException as e:
            print(e)
            sys.exit(1)

    def train(self,
              train_loader,
              wrapped_model
              ):
        torch.cuda.synchronize()
        m_type: str = self.config.strategy["model_type"]
        if m_type == "ddp":
            # 准备训练所需要的基本构建
            Epoch = self.config.train["epochs"]
            criterion = self._get_criterion(self.config.train["criterion"])
            optimizer = self._get_optim(self.config.train["optimizer"], wrapped_model)
            device = self.dist_args.local_rank
            gradient_step = self.config.train["gradient_accumulate_step"]
            iter_log = self.config.train["iter_log_interval"]
            epoch_log = self.config.train["epoch_log_interval"]

            # 模型进入训练状态
            wrapped_model.train()
            # 创建epoch_loss_list数组，用作以后的每一轮的loss的画图，比如折线图
            epoch_loss_list: List[float] = []

            for epoch in range(Epoch):  # loop for many epochs
                iter_loss = 0.0
                iters_loss = 0.0
                epoch_loss = 0.0
                train_iter = iter(train_loader)
                train_loader.sampler.set_epoch(epoch)

                for iter_number in tqdm(range(self.max_epoch_iter)):

                    # global_rank_print(0, "The iter_number : " + str(iter_number) + " of epoch :" + str(epoch+1))
                    if iter_number < self.epoch_iters[self.dist_args.global_rank]:

                        data_input, data_label = next(train_iter)
                        # 保存dummy sample
                        self._save_dummy_sample(iter_number, data_input, data_label)
                        # 开始真正的训练
                        data_input, data_label = to_device(data_input, device), to_device(data_label, device)
                        if iter_number == 0:  # tes whether the dataloader is right or not.
                            print("The one input is ", data_input)
                        with self._model_with_sync(wrapped_model, iter_number, gradient_step):
                            # 前向传播
                            data_output = wrapped_model(data_input)
                            # 计算loss值
                            iter_loss = criterion(data_output, data_label)
                            (iter_loss * self.grad_portion[self.dist_args.global_rank]).backward()

                        if (iter_number + 1) % gradient_step == 0:
                            optimizer.step()
                            optimizer.zero_grad()

                            # 开始假的训练
                        iters_loss += iter_loss.item()
                        epoch_loss += iter_loss.item()

                        if iter_log > 0 and iter_number % iter_log == (iter_log - 1):
                            print('[RANK %d][EPOCH %d][INDEX %d] : multiply iters average loss: %.4f' % (
                                self.dist_args.global_rank, epoch + 1, iter_number + 1, iters_loss / iter_log))
                            iters_loss = 0.0

                    else:  # choose a dummy example to manipulate the iter of small number process.
                        dummy_input = self.dummy_input.to(device)
                        dummy_label = self.dummy_label.to(device)

                        if epoch == 0:  # tes whether the dataloader is right or not.
                            print("The input shape is ", dummy_input.shape)

                        with self._model_with_sync(wrapped_model, iter_number, gradient_step):
                            # 前向传播
                            dummy_output = wrapped_model(dummy_input)
                            # 计算loss值
                            dummy_iter_loss = criterion(dummy_output, dummy_label)

                            dummy_iter_loss *= 0.0
                            dummy_iter_loss.backward()

                        if (iter_number + 1) % gradient_step == 0:
                            optimizer.step()
                            optimizer.zero_grad()

                if epoch_log > 0 and epoch % epoch_log == (epoch_log - 1):
                    print('[RANK %d][EPOCH %d] :Epoch average loss: %.4f' % (
                        self.dist_args.global_rank, epoch + 1, epoch_loss / epoch_log))

                    epoch_loss_list.append(epoch_loss)
                    epoch_loss = 0.0

        return wrapped_model, epoch_loss_list

    # 下面都是一些辅助的小函数
    ####################################################################################################################
    def _save_dummy_sample(self, iter_number, data_input, data_label):
        if iter_number == 0:
            self.dummy_input = data_input
            self.dummy_label = data_label

    def _model_with_sync(self, model, iter_index, gradient_accumulate_step):
        # from https://github.com/yifding/hetseq/blob/master/hetseq/controller.py
        if (hasattr(model, 'no_sync') and (iter_index + 1) % gradient_accumulate_step != 0):
            return model.no_sync()
        else:
            return contextlib.ExitStack()  # dummy contextmanager

    @staticmethod
    def get_parameter(model) -> dict:
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    def time_start(self, local_rank, target_local_rank):
        if local_rank == target_local_rank:
            self.start_time = time.time()

    def time_end(self, local_rank, target_local_rank):
        if local_rank == target_local_rank:
            self.end_time = time.time()

    def calculate_time(self, local_rank, target_local_rank):
        if local_rank == target_local_rank:
            return self.end_time - self.start_time

    # 通过名字创建相应的优化器
    # 假定模型已经在gpu上了
    def _get_optim(self, optim_name: str, model):
        try:
            if optim_name == "Adam":
                return optim.Adam(model.parameters(), self.config.train["learning_rate"])
            else:
                raise MyException("Error [fastspeed:_get_optim func]:No such optimizer named ", optim_name)
        except MyException as e:
            print(e)
            sys.exit(1)

    # 通过名字返回相应的loss function
    def _get_criterion(self, criterion_name: str):
        try:
            if criterion_name == "CrossEntropy":
                return nn.CrossEntropyLoss()
            else:
                raise MyException("Error [fastspeed:_get_criterion func]:No such criterion named ", criterion_name)
        except MyException as e:
            print(e)
            sys.exit(1)





