import pandas as pd
train = pd.read_csv("temporary_datasets/mnist_train.csv")
splitup_dataset = {}

for label in train.label.value_counts().index:
    splitup_dataset[label] = train[train.label==label]

max_sample_size = min([len(x) for x in splitup_dataset.values()])

number_of_clients = 5
sample_per_class_per_client = max_sample_size//number_of_clients

for client in range(number_of_clients):
    clientdf = pd.concat([df.sample(sample_per_class_per_client) for df in splitup_dataset.values()],ignore_index=True)
    clientdf = clientdf.sample(frac=1)
    print(clientdf.label.value_counts())
    print(clientdf.head())
    clientdf.to_csv(f"temporary_datasets/client-{client}-train.csv",index=False)