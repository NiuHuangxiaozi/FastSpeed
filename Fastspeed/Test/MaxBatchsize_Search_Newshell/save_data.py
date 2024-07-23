from typing import List
import torch
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer

def load_data(jsonl_file_path)->pd.DataFrame:

    #准备数据
    data_list=[]
    with open(jsonl_file_path, "r") as file:
        for line in file:
            json_obj = json.loads(line)
            data_list.append([json_obj['text'],json_obj['label']])
    df=pd.DataFrame(data_list,columns=['text','label'])
    print(len(df))
    return df

def split_data(df)->List[pd.DataFrame]:
    np.random.seed(112)
    df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42),[int(.97* len(df)), int(.99 * len(df))])
    print(len(df_train), len(df_val), len(df_test))
    return  df_train, df_val, df_test

class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self,tokenizer,df):
        self.labels = [torch.LongTensor([label]) for label in tqdm(df['label'])]
        self.texts = [tokenizer(text,
                                padding='max_length',
                                max_length=512,
                                truncation=True,
                                return_tensors="pt")
                      for text in tqdm(df['text'])
                      ]
    #返回数据的标签
    def classes(self):
        return self.labels

    #返回一共有几个标签
    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return self.labels[idx]

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y



def emotion_save():
    df_data = load_data("Data/data.jsonl")
    df_train, df_val, df_test = split_data(df_data)
    tokenizer = BertTokenizer.from_pretrained("./Config")
    test_dataset = EmotionDataset(tokenizer, df_test)
    text, label = test_dataset[0]
    input = \
        {
            'attention_mask': text['attention_mask'],
            'input_ids': text['input_ids']
        }
    input_and_label = {
        "input": input,
        "label": label
    }
    PATH = "./Data/testsample.pt"
    torch.save(input_and_label, PATH)

######################################################################################


import torchvision.transforms as transforms
import torchvision
def get_dataset(dl_path):
    #图片的数据处理
    transform = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(449),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

    trainset = torchvision.datasets.CIFAR10(root=dl_path,
                                            train=True,
                                            download=True,
                                            transform=transform)
    testset = torchvision.datasets.CIFAR10(root=dl_path,
                                           train=False,
                                           download=True,
                                           transform=transform)

    return trainset, testset



def cifa10_save():
    train_dataset, testset=get_dataset("./")
    input = train_dataset[0][0].unsqueeze(0)
    label = torch.Tensor([train_dataset[0][1]])
    PATH = "./Data/cifar_testsample.pt"
    input_and_label = {
        "input": input,
        "label": label
    }
    torch.save(input_and_label, PATH)
def main():
    #emotion_save()
    cifa10_save()
if __name__=="__main__":
    main()