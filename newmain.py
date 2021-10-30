import numpy as np
from sklearn.preprocessing import normalize as sk_normalize
from scipy import interpolate as scipy_interpolate
from keras import models, layers
import sys
import pickle
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import animation
import pandas


def down_sample(arr: np.ndarray, factor: int) -> np.ndarray:
    shape = (arr.shape[0], arr.shape[1] // factor)
    tmp = np.zeros(shape)
    for run in range(arr.shape[0]):
        tmp[run, :] = np.mean(arr[run, :].reshape(-1, factor), 1)

    return tmp


def interpolate(arr: np.ndarray, factor) -> np.ndarray:
    x = np.array(range(arr.shape[0]))
    y = np.array(range(arr.shape[1]))
    f = scipy_interpolate.interp2d(x, y, arr.T)

    xnew = np.linspace(0, arr.shape[0], arr.shape[0])
    ynew = np.linspace(0, arr.shape[1], arr.shape[1] * factor)
    return f(xnew, ynew).T


def normalize(arr: np.ndarray) -> np.ndarray:
    return sk_normalize(arr, axis=1, norm="max")  # can try l1 or l2 insted


def get_data(mode=0) -> np.ndarray:
    if mode == 0:
        ps1 = normalize(down_sample(np.loadtxt("data/PS1.txt"), 10))
        ps2 = normalize(down_sample(np.loadtxt("data/PS2.txt"), 10))
        ps3 = normalize(down_sample(np.loadtxt("data/PS3.txt"), 10))
        # ps4 = normalize(down_sample(np.loadtxt("data/PS4.txt"), 10)) # all values ar 0
        ps5 = normalize(down_sample(np.loadtxt("data/PS5.txt"), 10))
        ps6 = normalize(down_sample(np.loadtxt("data/PS6.txt"), 10))

        ts1 = normalize(interpolate(np.loadtxt("data/TS1.txt"), 10))
        ts2 = normalize(interpolate(np.loadtxt("data/TS2.txt"), 10))
        ts3 = normalize(interpolate(np.loadtxt("data/TS3.txt"), 10))
        ts4 = normalize(interpolate(np.loadtxt("data/TS4.txt"), 10))

        fs1 = normalize(np.loadtxt("data/FS1.txt"))
        fs2 = normalize(np.loadtxt("data/FS2.txt"))

        eps1 = normalize(down_sample(np.loadtxt("data/EPS1.txt"), 10))

        se = normalize(interpolate(np.loadtxt("data/SE.txt"), 10))

        ce = normalize(interpolate(np.loadtxt("data/CE.txt"), 10))

        cp = normalize(interpolate(np.loadtxt("data/CP.txt"), 10))

        # arr = np.array([ps1, ps2, ps3, ps5, ps6, ts1, ts2, ts3, ts4, fs1, fs2, eps1, se, ce, cp]).transpose((1, 0, 2))
        # arr = np.array([ps1, ps2, ps3, ps5, ps6]).transpose((1, 0, 2))
        arr = np.array([fs1, fs2]).transpose((1, 0, 2))
        with open("data_parsed/yoloswag", "wb") as f:
            pickle.dump(arr, f)
        return arr
    else:
        with open("data_parsed/yoloswag", "rb") as f:
            arr = pickle.load(f)
        return arr


def get_target_leakege() -> np.ndarray:
    tar = np.loadtxt("data/profile.txt")
    tar = tar[:, 2]
    arr = np.zeros((tar.shape[0], 3))
    for i in range(tar.shape[0]):
        if tar[i] == 0:
            arr[i, 0] = 1
        elif tar[i] == 1:
            arr[i, 1] = 1
        elif tar[i] == 2:
            arr[i, 2] = 1
        else:
            print("hvis du ser dette er noe feil!")
    return arr


def get_target_valve() -> np.ndarray:
    tar = np.loadtxt("data/profile.txt")
    tar = tar[:, 1]
    arr = np.zeros((tar.shape[0], 4))
    for i in range(tar.shape[0]):
        if tar[i] == 100:
            arr[i, 0] = 1
        elif tar[i] == 90:
            arr[i, 1] = 1
        elif tar[i] == 80:
            arr[i, 2] = 1
        elif tar[i] == 73:
            arr[i, 3] = 1
        else:
            print("hvis du ser dette er noe feil!")
    return arr


def get_target_hydraulic() -> np.ndarray:
    tar = np.loadtxt("data/profile.txt")
    tar = tar[:, 3]
    arr = np.zeros((tar.shape[0], 4))
    for i in range(tar.shape[0]):
        if tar[i] == 130:
            arr[i, 0] = 1
        elif tar[i] == 115:
            arr[i, 1] = 1
        elif tar[i] == 100:
            arr[i, 2] = 1
        elif tar[i] == 90:
            arr[i, 3] = 1
        else:
            print("hvis du ser dette er noe feil!")
    return arr


def neaural_pressure_valve():
    data = get_data(1)
    target = get_target_valve()

    p = np.random.permutation(data.shape[0])
    data = data[p, :, :]
    target = target[p, :]
    print(data.shape)
    train_data = data[200:-1, :, :]
    test_data = data[0:200, :, :]

    train_target = target[200:-1, :]
    test_target = target[0:200, :]

    train_data = train_data.reshape((train_data.shape[0], data.shape[1] * data.shape[2]))
    test_data = test_data.reshape((test_data.shape[0], data.shape[1] * data.shape[2]))

    network = models.Sequential()
    network.add(layers.Dense(2048, activation="relu", input_shape=(data.shape[1] * data.shape[2],)))
    # network.add(layers.Dropout(0.1))
    # network.add(layers.Dense(8, activation="relu"))
    network.add(layers.Dense(target.shape[1], activation="softmax"))
    network.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
    network.fit(train_data, train_target, epochs=30, batch_size=10)

    test_loss, test_acc = network.evaluate(test_data, test_target)
    print('test_acc: ', test_acc)

    y_pred = network.predict_classes(test_data)

    y_true = np.argmax(test_target, axis=1)
    maitrix = confusion_matrix(y_true, y_pred)
    labels = ['0', '1', '2', "3"]
    print('maitrix: ', maitrix)

    axis = plt.subplot()
    sns.heatmap(maitrix, annot=True, ax=axis, square=True, fmt='g')

    axis.set_xlabel('Predicted')
    axis.set_ylabel('True')
    axis.set_title('Confusion Matrix')
    axis.xaxis.set_ticklabels(labels)
    axis.yaxis.set_ticklabels(labels)
    plt.show()


def neaural_flow_valve():
    data = get_data(1)
    target = get_target_valve()

    p = np.random.permutation(data.shape[0])
    data = data[p, :, :]
    target = target[p, :]
    print(data.shape)
    train_data = data[200:-1, :, :]
    test_data = data[0:200, :, :]

    train_target = target[200:-1, :]
    test_target = target[0:200, :]

    train_data = train_data.reshape((train_data.shape[0], data.shape[1] * data.shape[2]))
    test_data = test_data.reshape((test_data.shape[0], data.shape[1] * data.shape[2]))

    network = models.Sequential()
    network.add(layers.Dense(4096, activation="relu", input_shape=(data.shape[1] * data.shape[2],)))
    # network.add(layers.Dropout(0.1))
    # network.add(layers.Dense(2048, activation="relu"))
    network.add(layers.Dense(target.shape[1], activation="softmax"))
    network.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
    network.fit(train_data, train_target, epochs=30, batch_size=10)

    test_loss, test_acc = network.evaluate(test_data, test_target)
    print('test_acc: ', test_acc)

    y_pred = network.predict_classes(test_data)

    y_true = np.argmax(test_target, axis=1)
    maitrix = confusion_matrix(y_true, y_pred)
    labels = ['0', '1', '2', "3"]
    print('maitrix: ', maitrix)

    axis = plt.subplot()
    sns.heatmap(maitrix, annot=True, ax=axis, square=True, fmt='g')

    axis.set_xlabel('Predicted')
    axis.set_ylabel('True')
    axis.set_title('Confusion Matrix')
    axis.xaxis.set_ticklabels(labels)
    axis.yaxis.set_ticklabels(labels)
    plt.show()


def reg():
    ps1 = normalize((np.loadtxt(f"data/FS1.txt")))

    # tmp = np.zeros(ps1.shape)
    # for run in range(ps1.shape[0]):
    #     for i in range(ps1.shape[1] - 1):
    #         tmp[run, i] = ps1[run, i + 1] - ps1[run, i]

    # tmp = np.abs(tmp)
    ps1 = ps1[:, 120:]
    s = np.sum(ps1, axis=1)

    tar = np.loadtxt("data/profile.txt")
    tar = tar[:, 2]

    # plt.scatter(tar, s)

    tar = tar.reshape(1, -1)
    s = s.reshape(1, -1)
    lin_reg = LinearRegression()
    lin_reg.fit(s, tar)
    print(lin_reg.score(s, tar))


if __name__ == '__main__':
    # neaural_flow_valve()

    reg()
