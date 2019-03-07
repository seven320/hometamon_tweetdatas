# encoding:utf-8
import copy
import numpy as np
import random
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LSTM(nn.Module):
    def __init__(self,data_size,hidden_size,output_size):
        super(LSTM,self).__init__()
        self.output_size = output_size
        #
        self.lstm = nn.LSTM(data_size,hidden_size,batch_first=True)
        self.hidden2output = nn.Linear(hidden_size,output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim = 1)
        self.optimizer = optim.RMSprop(self.parameters(), lr=0.00015, alpha=0.95, eps=0.01)

    def __call__(self, inputs, hidden = None):
        #lstmではtorchのモデルを使えるので入力に全入力情報を一度に渡せる
        # print("0",inputs.shape)
        outputs, (hn,cell) = self.lstm(inputs,hidden)
        # print("1",outputs.shape)
        output = self.hidden2output(outputs[-1,:,:])#最後の層だけ取り出し
        # print("2",output.shape)
        output = self.softmax(output)
        # print("3",output.shape)
        return output
