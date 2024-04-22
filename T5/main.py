# official lib
import os
import argparse
import torch
from torch.distributed import init_process_group, destroy_process_group
from argparse import Namespace
from transformers import BertTokenizer
import json
import pandas as pd
from transformers import T5Tokenizer
# my lib
from Fastspeed.utils.utilstool import Params
from Fastspeed.utils.fastspeed import FastSpeed

from load_data import SummaryDataset
from model import T5,getT5calculationCost

def Get_args():
    parser = argparse.ArgumentParser(description='Alexnet train on cifar10.')
    parser.add_argument('--json_path', default="./args.json", help="args.json file path")
    parser.add_argument('--local_rank', default=-1, type=int, help='Local rank always refer to specific gpu.')
    parser.add_argument('--global_rank', default=-1, type=int, help='Global Rank.')
    parser.add_argument('--world_size', default=-1, type=int, help='All ranks.')

    # more like to change
    ##############################
    parser.add_argument('--modelType', type=str, default="T5base", help="The type of T5(small,base,large,...)")
    parser.add_argument('--configPath', type=str, default="./T5base_config/", help="the path of model config")

    ####################################
    parser.add_argument('--maxLen', default=512, type=int, help='the max length of input')
    parser.add_argument('--summaryLen', default=150, type=int, help='the max length of output')
    parser.add_argument('--showPreds', default=False, type=bool, help='whether to print every epcoch summary result')
    ####################################################################################################################

    args = parser.parse_args()
    #获得分布式训练的本地rank和全局的rank。
    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.global_rank = int(os.environ["RANK"])
    args.world_size = int(os.environ['WORLD_SIZE'])
    return args

def Distributed_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
def Distributed_destroy():
    destroy_process_group()

def main():
    Distributed_setup()
    args = Get_args()
    param = Params(args.json_path)
    if args.local_rank == 0:
        print("The config is :", json.dumps(vars(param), indent=4))
    Task(args,param)
    print("RANK :", args.global_rank, "All Finished")
    Distributed_destroy()






def Task(
        args:Namespace,
        param:Params
         ):
    #加载数据
    print("Begin to load news_summary data")

    Data_path = './Data/news_summary.csv'

    df = pd.read_csv(Data_path, encoding='latin-1')
    df = df[['text', 'ctext']]
    print("All data :", df)
    # 划分训练集和测试集
    propotion = 0.2
    train_df = df.sample(frac=propotion, random_state=0)
    test_df = df.drop(train_df.index).reset_index(drop=True)
    train_df = train_df.reset_index(drop=True)
    print("FULL Dataset shape : {}".format(df.shape))
    print("TRAIN Dataset shape: {}".format(train_df.shape))
    print("VAL Dataset shape : {}".format(test_df.shape))

    tokenizer = T5Tokenizer.from_pretrained(args.configPath, legacy=False)

    # 构建数据集
    train_dataset = SummaryDataset(train_df, tokenizer, args.maxLen, args.summaryLen)

    # 取一个样例出来，查看相应的数据和形状
    input,label = train_dataset[0]
    #加上batchsize的维度
    for key in input:
        input[key]=input[key].reshape(1,-1)

    print(f"source_ids shape {input['source_ids'].shape}")
    print(f"source_ids {input['source_ids']}")

    print(f"source_mask shape {input['source_mask'].shape}")
    print(f"source_mask {input['source_mask']}")

    print(f"target_ids shape {input['target_ids'].shape}")
    print(f"target_ids {input['target_ids']}")

    print(f"target_ids_y shape {input['target_ids_y'].shape}")
    print(f"target_ids_y {input['target_ids_y']}")

    # 定义模型
    task_model = T5(args.configPath)

    # 定义异构训练平台
    train_platform = FastSpeed(param, args, input, label,True)

    # 平台进行异构的数据划分
    train_loader = train_platform.unbalanced_datasplit(dataset=train_dataset, model=task_model)

    # 统计模型参数量
    if args.local_rank==0:
        print(f"The {args.modelType} model's total parameter is {train_platform.get_parameter(task_model)}")

        print('In task on this node, we use {} gpus!'.format(torch.cuda.device_count()))

    # 计算单一sample需要的浮点数运算次数
    T5CalculationQuantity = getT5calculationCost(args.configPath+"/config.json",args.maxLen)

    # 加载模型，开始训练
    wrapped_model = train_platform.model_wrap(task_model)
    trained_model, epoch_loss_list, timeCost, throughput, totalThroughput = train_platform.train(
        train_loader=train_loader, wrapped_model=wrapped_model)

    print(
        f"[GLOBAL_RANK:{args.global_rank}]  [Epoch_loss_list:{epoch_loss_list}]   [TimeCost:{timeCost}s]"
        f"[Throughput:{throughput} sample/s] [TFLOPS: {(throughput * T5CalculationQuantity) / 1e12}]")


    print(f"Fastspeed Total statistics [Throughput:{totalThroughput} sample/s] "
          f"[TFLOPS per GPU: {(totalThroughput * T5CalculationQuantity) / (args.world_size * 1e12)}]")

    print('Global Rank %d finished training' % (args.global_rank))









# /////////////////////////////////////////////////////////////////
if __name__ == '__main__':
    main()

