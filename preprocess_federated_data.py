import pandas as pd
import numpy as np
import os

def load_data(index):
    df = pd.read_csv(f"./fedavgdatasets/Task1_W_Zone{index}.csv")
    columns = df.columns
    df.fillna(df.mean(), inplace=True)
    for i in range(3, 7):
        MAX = np.max(df[columns[i]])
        MIN = np.min(df[columns[i]])
        df[columns[i]] = (df[columns[i]] - MIN) / (MAX - MIN)

    return df

def nn_seq_wind(index,B):
    print('data processing...')
    dataset = load_data(index) 
    # split
    train = dataset[:int(len(dataset) * 0.6)]
    val = dataset[int(len(dataset) * 0.6):int(len(dataset) * 0.8)]
    test = dataset[int(len(dataset) * 0.8):len(dataset)]

    def process(data):
        columns = data.columns
        wind = data[columns[2]]
        wind = wind.tolist()
        data = data.values.tolist()
        X, Y = [], []
        for i in range(len(data) - 30):
            train_seq = []
            train_label = []
            for j in range(i, i + 24):
                train_seq.append(wind[j])

            for c in range(3, 7):
                train_seq.append(data[i + 24][c])
            train_label.append(wind[i + 24])
            X.append(train_seq)
            Y.append(train_label)

        X, Y = np.array(X), np.array(Y)

        length = int(len(X) / B) * B
        X, Y = X[:length], Y[:length]

        return X, Y

    train_x, train_y = process(train)
    val_x, val_y = process(val)
    test_x, test_y = process(test)

    return [train_x, train_y], [val_x, val_y], [test_x, test_y]

if __name__=='__main__':
    batch_size = 32
    for i in range(1,11):
        tr,val,test = nn_seq_wind(i,batch_size)
        
        print("*"*10)
        training_data = pd.DataFrame(tr[0])
        training_data["y"] = pd.DataFrame(tr[1])
        print(training_data.shape)
        print(f"fed_training_data{i}.csv")
        training_data.to_csv(f"fed_training_data{i}.csv")

        print("*"*10)
        validation_data = pd.DataFrame(val[0])
        validation_data["y"] = pd.DataFrame(val[1])
        print(validation_data.shape)
        print(f"fed_validation_data{i}.csv")
        validation_data.to_csv(f"fed_validation_data{i}.csv")

        print("*"*10)
        testing_data = pd.DataFrame(test[0])
        testing_data["y"] = pd.DataFrame(test[1])
        print(testing_data.shape)
        print(f"fed_testing_data{i}.csv")
        testing_data.to_csv(f"fed_testing_data{i}.csv")