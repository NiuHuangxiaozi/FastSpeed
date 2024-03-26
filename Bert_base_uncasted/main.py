# official lib

import torch
from torch.distributed import init_process_group, destroy_process_group
import os
import argparse
from argparse import Namespace
from transformers import BertTokenizer

# my lib
from Fastspeed.utils.utilstool import Params
from Fastspeed.utils.fastspeed import FastSpeed
from load_data import load_data,split_data,EmotionDataset
from model import BertClassifier
def Get_args():
    parser = argparse.ArgumentParser(description='Alexnet train on cifar10.')
    parser.add_argument('--json_path', default="./args.json", help="args.json file path")
    parser.add_argument('--local_rank', default=-1, type=int, help='Local rank always refer to specific gpu.')
    parser.add_argument('--global_rank', default=-1, type=int, help='Global Rank.')
    parser.add_argument('--world_size', default=-1, type=int, help='All ranks.')
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
        print("The config is :", vars(param))
    Task(args,param)
    print("RANK :", args.global_rank, "All Finished")
    Distributed_destroy()


def Task(
        args:Namespace,
        param:Params
         ):
    print("Begin to load data.")
    DATA_PATH="Data/data.jsonl"
    TOKENIZER_PATH="./Config/"

    df = load_data(DATA_PATH)
    df_train, df_val, df_test = split_data(df)
    tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)
    #train_data = EmotionDataset(tokenizer, df_train)
    val_data = EmotionDataset(tokenizer, df_val)

    #获取数据集和用于负载均衡的小样本
    input,label=val_data[0]
    input = {
                    'attention_mask': input["attention_mask"],
                    'input_ids':input["input_ids"].reshape(1,-1)
            }
    label=torch.LongTensor([label])

    # 定义模型
    task_model = BertClassifier(TOKENIZER_PATH)
    #定义异构训练平台
    train_platform=FastSpeed(param,args,input,label)
    #平台进行异构的数据划分
    train_loader = train_platform.unbalanced_datasplit(dataset=val_data,model=task_model)

    if args.local_rank==0:
        print("The model's total parameter is ",train_platform.get_parameter(task_model))
        print('In task on this node, we use {} gpus!'.format(torch.cuda.device_count()))

    #加载模型，开始训练
    wrapped_model = train_platform.model_wrap(task_model)
    trained_model,epoch_loss_list=train_platform.train(train_loader=train_loader,
                                                wrapped_model=wrapped_model)

    print("epoch_loss_list is",epoch_loss_list)
    print('Global Rank %d finished training' % (args.global_rank))

# /////////////////////////////////////////////////////////////////
if __name__ == '__main__':
    main()

