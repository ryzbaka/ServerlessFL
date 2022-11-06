import os
import pandas as pd

train_data = pd.read_csv("temporary_datasets/mnist_train.csv")
train_data.sort_values(by=["label"],inplace=True)
n =2 
non_iid_data=[
train_data[0:int(len(train_data)/n)],
train_data[int(len(train_data)/n):int(2*len(train_data)/n)],
# train_data[int(2*len(train_data)/n):int(3*len(train_data)/n)],
]
for i in range(len(non_iid_data)):
    df = non_iid_data[i]
    filename = f"mnist_non_iid/non_iid_{i}.csv"
    print(df.label.value_counts())
    df.to_csv(filename)