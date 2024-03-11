# 自己写的其他的库
from utils.sampler import Distributed_Elastic_Sampler
from help.debug import global_rank_print
from utils.calculation import calculate_time, calculate_param
# exception
from help.debug import MyException

# 官方的库
import contextlib
import copy
import sys
import torch
import time
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import List, Union
import torch.distributed as dist

'''
这个文件的作用是实现数据集的合理划分以及模型流水线并行的合理划分
'''
__all__ = ["OptimusSpeed", ]


class OptimusSpeed:

    def __init__(self) -> None:
        self.grad_portion: List[float] = None  # 对于数据并行不平衡的batch的normalize
        # 用来计算模型的运行时间并做一些分析
        self.start_time = 0.0
        self.end_time = 0.0

        # 统一每一个epoch的iter数，防止有些节点训练完毕，有些没有训练完毕
        self.max_epoch_iter: int = 0
        self.epoch_iters: List[int] = []
        self.dummy_input = None  # 以下两个参数，用作在iter不够的时候顶上去
        self.dummy_label = None

    '''
    model_wrap:构建我们需要训练的模型
    '''

    def model_wrap(self, model, device, model_type: str):
        try:
            if model_type == 'ddp':
                self.wrapped_model = DDP(model.to(device))
            elif model_type == 'pipe':
                pass
            elif model_type == 'hybrid':
                pass
            else:
                raise ValueError
        except ValueError:
            print("Current model parallel type only contain ddp, pipe and hybrid(ddp,pipe)")
            sys.exit(1)
        return self.wrapped_model

    def _autobalance(self,
                     model,
                     world_size,
                     criterion,
                     config_param,
                     args
                     ) -> Union[List[int], List[int]]:
        partition_list = None
        batchsize_list = None
        try:
            if config_param.model_type == 'ddp':
                dummy_model = copy.deepcopy(model)  # 独立拷贝一份模型用作测试防止修改原来的模型
                if config_param.unbalanced_strategy == "time":

                    dummy_time = calculate_time(dummy_model, criterion, args.local_rank, config_param.data_dummy_shape)
                    dummy_time_tensor = torch.tensor(dummy_time).to(args.local_rank)
                    time_list = [torch.zeros(1, ).to(args.local_rank) for i in range(0, world_size)]
                    print("time_list", time_list)
                    print("dummy_time_tensor", dummy_time_tensor)
                    dist.all_gather(time_list, dummy_time_tensor)
                    print("time_list", time_list)
                    time_list = [obj.item() * 100 for obj in time_list]
                    rounded_time_array = np.trunc(time_list) / 100
                    # rounded_time_array = (np.rint(time_list)).astype(int)
                    print("rounded_time_array", rounded_time_array)
                    portion = rounded_time_array / np.sum(rounded_time_array)
                    print("portion", portion)
                    partition_list = (config_param.total_datasize * portion).astype(int)
                    batchsize_list = (config_param.batch_size * portion).astype(int)
                    partition_list[-1] = config_param.total_datasize - np.sum(partition_list[:-1])
                    batchsize_list[-1] = config_param.batch_size - np.sum(batchsize_list[:-1])
                    return partition_list.tolist(), batchsize_list.tolist()
                else:
                    raise MyException(2, "The auto balanced strategy is only support for time and complexity now.")
            elif config_param.model_type == 'pipe':
                pass
            else:
                raise MyException(3, "The parallel type only support for ddp ,pipe or hybrid")
        except MyException as e:
            print(e)
            sys.exit(1)

    '''
    optimus.ddp_unbalanced_dataset_split(
                                        dataset=train_dataset,
                                        model=task_model,
                                        criterion=criterion,
                                        config_param=param,
                                        args=args)
    '''
    def ddp_unbalanced_dataset_split(
            self,
            dataset,
            model,
            criterion,
            config_param,
            args
    ) -> torch.utils.data.DataLoader:
        try:
            if config_param.partition_method == 'manual' or config_param.partition_method == 'autobalanced':

                #这一段选择我们的partition_list和batchsize_list是从哪里来，是人工还是系统自己计算
                if config_param.partition_method == 'autobalanced':
                    print("This is autobalanced partition mode!")
                    partition_list, batchsize_list = self._autobalance(model=model, world_size=args.world_size,
                                                                       criterion=criterion, config_param=config_param,
                                                                       args=args)
                else:
                    print("This is manual partition mode!")
                    partition_list,batchsize_list=config_param.manual_partition_list,config_param.manual_batchsize_list


                # 统一每一个epoch的iter数，防止有些节点训练完毕，有些没有训练完毕
                self.epoch_iters = np.ceil(np.divide(partition_list, batchsize_list)).astype(int)
                self.max_epoch_iter = max(self.epoch_iters)
                global_rank_print(0, "epoch_iters is " + str(list(self.epoch_iters)))
                global_rank_print(0, "max_epoch_iter is " + str(self.max_epoch_iter))

                # for debug
                global_rank_print(0, "The partition_list is : " + str(partition_list))
                global_rank_print(0, "The batchsize_list is : " + str(batchsize_list))

                # for debug
                if partition_list is None or batchsize_list is None:
                    raise MyException(0, "partition_list or batchsize_list can't be None in the manual mode!")

                # 这里在计算完batchsize_list之后我们需要计算一下各个梯度的权重比例
                self.grad_portion = batchsize_list / np.sum(batchsize_list)

                sampler_dict = \
                    {
                        'method': "uneven",
                        'partition_list': partition_list
                    }

                sampler = Distributed_Elastic_Sampler(dataset=dataset, partition_strategy=sampler_dict)
                train_loader = DataLoader(dataset=dataset,
                                          batch_size=batchsize_list[args.global_rank],
                                          shuffle=False,  # 这个值必须设置为false，否则会导致多个节点可能都抽到同一个样例的结果
                                          sampler=sampler,
                                          pin_memory=config_param.train_loader_pin_memory,
                                          num_workers=config_param.train_loader_num_workers
                                          )
                return train_loader
            else:
                raise MyException(1, "Your choice of partition_method(" + str(
                    config_param.partition_method) + ") is wrong [neither manual or autobalanced!")
        except MyException as e:
            print(e)
            sys.exit(1)

    def train(self,
              train_loader,
              wrapped_model,
              criterion,
              optimizer,
              param,
              args
              ):

        if param.model_type == "ddp":
            # 模型进入训练状态
            wrapped_model.train()
            # 创建epoch_loss_list数组，用作以后的每一轮的loss的画图，比如折线图
            epoch_loss_list: List[float] = []
            for epoch in range(param.epochs):  # loop for many epochs

                iter_loss = 0.0
                iters_loss = 0.0
                epoch_loss = 0.0
                train_iter = iter(train_loader)
                train_loader.sampler.set_epoch(epoch)

                for iter_number in range(self.max_epoch_iter):

                    global_rank_print(0, "The iter_number : " + str(iter_number) + " of epoch :" + str(epoch+1))
                    if iter_number < self.epoch_iters[args.global_rank]:

                        data_input, data_label = next(train_iter)
                        # 保存dummy sample
                        self._save_dummy_sample(iter_number, data_input, data_label)
                        # 开始真正的训练
                        data_input, data_label = data_input.to(args.local_rank), data_label.to(args.local_rank)
                        if iter_number == 0:  # tes whether the dataloader is right or not.
                            print("The input shape is ", data_input.shape)
                        with self._model_with_sync(wrapped_model, iter_number, param.gradient_accumulate_step):
                            # 前向传播
                            data_output = wrapped_model(data_input)
                            # 计算loss值
                            iter_loss = criterion(data_output, data_label)
                            (iter_loss * self.grad_portion[args.global_rank]).backward()

                        if (iter_number + 1) % param.gradient_accumulate_step == 0:
                            optimizer.step()
                            optimizer.zero_grad()

                            # 开始假的训练
                        iters_loss += iter_loss.item()
                        epoch_loss += iter_loss.item()

                        if param.iter_log_interval > 0 and iter_number % param.iter_log_interval == (
                                param.iter_log_interval - 1):
                            print('[RANK %d][EPOCH %d][INDEX %d] : multiply iters average loss: %.4f' % (
                                args.global_rank, epoch + 1, iter_number + 1, iters_loss / param.iter_log_interval))
                            iters_loss = 0.0

                    else:  # choose a dummy example to manipulate the iter of small number process.
                        dummy_input = self.dummy_input.to(args.local_rank)
                        dummy_label = self.dummy_label.to(args.local_rank)

                        if epoch == 0:  # tes whether the dataloader is right or not.
                            print("The input shape is ", dummy_input.shape)

                        with self._model_with_sync(wrapped_model, iter_number, param.gradient_accumulate_step):
                            # 前向传播
                            dummy_output = wrapped_model(dummy_input)
                            # 计算loss值
                            dummy_iter_loss = criterion(dummy_output, dummy_label)

                            dummy_iter_loss *= 0.0
                            dummy_iter_loss.backward()

                        if (iter_number + 1) % param.gradient_accumulate_step == 0:
                            optimizer.step()
                            optimizer.zero_grad()

                if param.epoch_log_interval > 0 and epoch % param.epoch_log_interval == (param.epoch_log_interval - 1):
                    print('[RANK %d][EPOCH %d] :Epoch average loss: %.4f' % (
                        args.global_rank, epoch + 1, epoch_loss / param.epoch_log_interval))

                    epoch_loss_list.append(epoch_loss)

                    epoch_loss = 0.0

        return wrapped_model, epoch_loss_list

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






