import json
import torch
import numpy as np
import pandas as pd
from typing import List
from tqdm import tqdm

EmotionLabels = {
    0: 'sadness',
    1: 'joy',
    2: 'love',
    3: 'anger',
    4: 'fear',
    5: 'surprise'
}


def load_data(jsonl_file_path) -> pd.DataFrame:
    # 准备数据
    data_list = []
    with open(jsonl_file_path, "r") as file:
        for line in file:
            json_obj = json.loads(line)
            data_list.append([json_obj['text'], json_obj['label']])
    df = pd.DataFrame(data_list, columns=['text', 'label'])
    print(len(df))
    return df


def split_data(df) -> List[pd.DataFrame]:
    np.random.seed(112)
    df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42), [int(.98 * len(df)), int(.99 * len(df))])
    print(len(df_train), len(df_val), len(df_test))
    return df_train, df_val, df_test


class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, df):
        print("load labels")
        self.labels = [label for label in tqdm(df['label'])]
        print("load texts")
        self.tokens = [tokenizer(text,
                                 padding='max_length',
                                 max_length=512,
                                 truncation=True,
                                 return_tensors="pt")
                       for text in tqdm(df['text'])
                       ]
        self.texts = [{'input_ids': token['input_ids'].reshape(-1), 'attention_mask': token['attention_mask']} for token
                      in self.tokens]

    # 返回数据的标签
    def classes(self):
        return self.labels

    # 返回一共有几个标签
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
        return (batch_texts, batch_y)

    # # seed init.
    # random.seed(seed)
    # np.random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    #
    # # torch seed init.
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    df = load_data("Data/data.jsonl", "Config/")
    df_train, df_val, df_test = split_data(df)