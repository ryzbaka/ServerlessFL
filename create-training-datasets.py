import os
import randomname
import pandas as pd
# check for existing client-datasets folder -> create if not exist, else delete all files in the folder
if "client-datasets" not in os.listdir():
    os.mkdir("client-datasets")
else:
    existing_client_files = os.listdir("client-datasets")
    for filename in existing_client_files:
        os.remove(f"client-datasets/{filename}")

#split data for clients
num_client = int(input("Enter the number of clients: "))
num_instances = int(input("Enter number of instances per class per client: "))
main_data = pd.read_csv("temporary_datasets/mnist_train.csv")
labels = list(main_data.label.value_counts().index)
class_datasets = [main_data[main_data.label==label_value] for label_value in labels] 
for i in range(num_client):
    client_name = "".join(randomname.get_name().split("-"))
    client_data = pd.concat([dataset.sample(num_instances) for dataset in class_datasets])
    client_data = client_data.sample(len(client_data))
    client_data.to_csv(f"client-datasets/client-{client_name}-train.csv",index=False)
    print(f"Created dataset: {client_name}-train.csv")
    print(client_data.label.value_counts())
    print("-"*10)

