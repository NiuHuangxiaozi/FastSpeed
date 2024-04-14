#import pandas as pd
from torch.utils.data import Dataset
import torch
#from transformers import T5Tokenizer

class SummaryDataset(Dataset):

    def __init__(self, dataframe, tokenizer, source_len, summ_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = summ_len
        self.text = self.data.text
        self.ctext = self.data.ctext

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        ctext = str(self.ctext[index])
        ctext = ' '.join(ctext.split())

        text = str(self.text[index])
        text = ' '.join(text.split())

        source = self.tokenizer.batch_encode_plus([ctext], max_length=self.source_len,  padding='max_length',truncation=True,
                                                  return_tensors='pt')
        target = self.tokenizer.batch_encode_plus([text], max_length=self.summ_len, padding='max_length',truncation=True,
                                                  return_tensors='pt')
        #print(f" before squeeze:source['input_ids']{source['input_ids'].shape}")
        source_ids = source['input_ids'].squeeze()
        #print(f" after squeeze:source['input_ids']{source_ids.shape}")

        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return (
            {
            'source_ids': source_ids.to(dtype=torch.long),
            'source_mask': source_mask.to(dtype=torch.long),
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
            },
            {}
        )







# def main():
#     TRAIN_BATCH_SIZE = 2
#     VALID_BATCH_SIZE = 2
#     TRAIN_EPOCHS = 5
#     VAL_EPOCHS = 1
#     LEARNING_RATE = 1e-4
#     MAX_LEN = 512
#     SUMMARY_LEN = 150
#
#     #读取基本的数据
#     df = pd.read_csv('./Data/news_summary.csv', encoding='latin-1')
#     df = df[['text', 'ctext']]
#     print(df)
#
#     train_size = 0.8
#     train_dataset = df.sample(frac=train_size, random_state=0)
#     val_dataset = df.drop(train_dataset.index).reset_index(drop=True)
#     train_dataset = train_dataset.reset_index(drop=True)
#
#     print("FULL Dataset: {}".format(df.shape))
#     print("TRAIN Dataset: {}".format(train_dataset.shape))
#     print("VAL Dataset: {}".format(val_dataset.shape))
#
#     t5_path="./T5config/"
#     tokenizer = T5Tokenizer.from_pretrained(t5_path, legacy=False)
#
#     #构建数据集
#     training_set = SummaryDataset(train_dataset, tokenizer, MAX_LEN, SUMMARY_LEN)
#
#     item=training_set[0]
#     print(item["source_ids"])
#     print(item["source_mask"])
#     print(item["target_ids"])
#     print(item["target_ids_y"])
#
#
#
#
#
# if __name__=="__main__":
#     main()
