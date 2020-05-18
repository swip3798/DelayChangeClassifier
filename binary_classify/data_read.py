import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler    
from sklearn.model_selection import train_test_split

from binary_classify.constants import TRAIN_BATCH_SIZE, TEST_BATCH_SIZE

def get_raw_dataframe(filename):
    df = pd.read_csv(filename)
    df.head()
    df.apply(pd.to_numeric)
    return df


def get_preprocessed_dataframe(filename = "delay_changed_l1.csv", delete_delay = True):
    df = get_raw_dataframe(filename)
    del df["id"]
    if delete_delay:
        del df["delay"]
    return df

def get_input_label():
    df = get_preprocessed_dataframe()
    inputs = df.iloc[:, :-1]
    labels = df.iloc[:, -1]
    return inputs, labels


def get_train_and_test():
    inputs, labels = get_input_label()
    input_train, input_test, label_train, label_test = train_test_split(inputs, labels, test_size=0.33, random_state=69)
    scaler = StandardScaler()
    input_train = scaler.fit_transform(input_train)
    input_test = scaler.fit_transform(input_test)
    return input_train, label_train, input_test, label_test

input_train, label_train, input_test, label_test = get_train_and_test()

## train data
class TrainData(Dataset):
    
    def __init__(self, input_data, label_data):
        self.input_data = input_data
        self.label_data = label_data
        
    def __getitem__(self, index):
        return self.input_data[index], self.label_data[index]
        
    def __len__ (self):
        return len(self.input_data)


train_data = TrainData(torch.FloatTensor(input_train), 
                       torch.FloatTensor(label_train))

## test data    
class TestData(Dataset):
    
    def __init__(self, input_data):
        self.input_data = input_data
        
    def __getitem__(self, index):
        return self.input_data[index]
        
    def __len__ (self):
        return len(self.input_data)
    

test_data = TestData(torch.FloatTensor(input_test))


train_loader = DataLoader(dataset=train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=TEST_BATCH_SIZE)

if __name__ == "__main__":
    df = get_preprocessed_dataframe()
    print(df)