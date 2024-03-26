#
# Implementation of AlexNet for illustrative purposes. The train.py driver
# can import AlexNet from here or directly from torchvision.
#
# Taken from torchvision.models.alexnet:
# https://pytorch.org/docs/1.6.0/_modules/torchvision/models/alexnet.html#alexnet

from torch import nn
from transformers import BertModel

class BertClassifier(nn.Module):
    def __init__(self, config_path,dropout=0.5):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(config_path)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 6)
        self.relu = nn.ReLU()

    def forward(self, input):

        _, pooled_output = self.bert(input_ids=input['input_ids'], attention_mask=input['attention_mask'],return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer