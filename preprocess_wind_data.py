import pandas as pd
import numpy as np

def load_data():
    df = pd.read_csv("wind_power_forecast.csv")
    columns = df.columns
    df.fillna(df.mean(), inplace=True)
    for i in range(3, 7):
        MAX = np.max(df[columns[i]])
        MIN = np.min(df[columns[i]])
        df[columns[i]] = (df[columns[i]] - MIN) / (MAX - MIN)

    return df


def nn_seq_wind(B):
    print('data processing...')
    dataset = load_data() 
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
    # df = load_data()
    # print(df.head())
    batch_size = 32
    tr, val, te = nn_seq_wind(batch_size)
    print("*"*10)
    print("training data information")
    print(type(tr))
    print(tr[0].shape)
    print(tr[1].shape)
    training_data = pd.DataFrame(tr[0])
    training_data["y"] = pd.DataFrame(tr[1])
    training_data.to_csv("training_data.csv")
    print(training_data.head())
    print("*"*10)
    print("validation data information")
    print(type(val))
    print(val[0].shape)
    print(val[1].shape)
    validation_data = pd.DataFrame(val[0])
    validation_data["y"] = pd.DataFrame(val[1])
    validation_data.to_csv("validation_data.csv")
    print("*"*10)
    print("testing data information")
    print(type(te))
    print(te[0].shape)
    print(te[1].shape)
    testing_data = pd.DataFrame(tr[0])
    testing_data["y"] = pd.DataFrame(tr[1])
    testing_data.to_csv("testing_data.csv")