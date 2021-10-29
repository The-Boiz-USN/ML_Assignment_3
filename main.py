import matplotlib.pyplot as plt
import numpy as np
from data_parser import get_data
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
import pickle
from keras import models, layers
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """
    return np.isnan(y), lambda z: z.nonzero()[0]


def plot_temperature_data(run: int):
    ts1 = get_data("ts1")[run, :]
    ts2 = get_data("ts2")[run, :]
    ts3 = get_data("ts3")[run, :]
    ts4 = get_data("ts4")[run, :]

    plt.figure("Temperature")
    plt.plot(ts1, label="ts1")
    plt.plot(ts2, label="ts2")
    plt.plot(ts3, label="ts3")
    plt.plot(ts4, label="ts4")
    plt.xlabel("Time [s]")
    plt.ylabel("C°")
    plt.legend()
    plt.grid()
    plt.title("Temperature")


def plot_efficiency_data(run: int):
    ce = get_data("ce")[run, :]
    se = get_data("se")[run, :]

    plt.figure("Efficiency")
    plt.title("Efficiency")
    plt.plot(ce, label="ce")
    plt.plot(se, label="se")
    plt.grid()
    plt.legend()
    plt.ylabel("%")
    plt.xlabel("Time [s]")


def plot_vibration_data(run: int):
    vs1 = get_data("vs1")[run, :]

    plt.figure("Vibration")
    plt.title("Vibration")
    plt.plot(vs1, label="vs1")
    plt.grid()
    plt.legend()
    plt.ylabel("mm/s")
    plt.xlabel("Time [s]")


def plot_power_data(run: int):
    cp = get_data("cp")[run, :]
    eps1 = get_data("eps1")[run, :]

    plt.figure("Power")
    plt.subplot(2, 1, 1)
    plt.title("Power")
    plt.plot(cp, label="cp")
    plt.grid()
    plt.legend()
    plt.ylabel("kW")

    plt.subplot(2, 1, 2)
    plt.plot(eps1, label="eps1")
    plt.grid()
    plt.legend()
    plt.ylabel("W")
    plt.xlabel("Time [s]")


def plot_pressure_data(run: int):
    ps1 = get_data("ps1")[run, :]
    ps2 = get_data("ps2")[run, :]
    ps3 = get_data("ps3")[run, :]
    ps4 = get_data("ps4")[run, :]
    ps5 = get_data("ps5")[run, :]
    ps6 = get_data("ps6")[run, :]

    plt.figure("Pressure")
    plt.subplot(3, 1, 1)
    plt.plot(ps1, label="ps1")
    plt.plot(ps2, label="ps2")
    plt.legend()
    plt.ylabel("Bar")
    plt.grid()
    plt.title("Pressure")

    plt.subplot(3, 1, 2)
    plt.plot(ps3, label="ps3")
    plt.plot(ps4, label="ps4")
    plt.legend()
    plt.ylabel("Bar")
    plt.grid()

    plt.subplot(3, 1, 3)
    plt.plot(ps5, label="ps5")
    plt.plot(ps6, label="ps6")
    plt.ylabel("Bar")
    plt.xlabel("Time (10 ms)")
    plt.legend()
    plt.grid()


def plot_flow_data(run: int):
    fs1 = get_data("fs1")[run, :]
    fs2 = get_data("fs2")[run, :]

    fig = plt.figure("Flow")
    plt.plot(fs1, label="fs1")
    plt.plot(fs2, label="fs2")
    plt.legend()
    plt.xlabel("Time [100 ms]")
    plt.ylabel("Flow [liter/min]")
    plt.grid()
    plt.title("Flow")


def plot(run: int):
    plot_pressure_data(run)
    plot_flow_data(run)
    plot_temperature_data(run)
    plot_efficiency_data(run)
    plot_vibration_data(run)
    plot_power_data(run)
    plt.show()


def crate_dataframe(data: list, columns: list) -> pd.DataFrame:
    tmp = np.array(data)
    tmp = tmp.transpose()
    df = pd.DataFrame(tmp, columns=columns)
    return df


def resize(data, hz, length=6000) -> np.ndarray:
    ts = 100 // hz

    tmp = np.empty(length)
    tmp[:] = np.nan
    for i in range(length):
        if i % ts == 0:
            tmp[i] = data[i // ts]
    return tmp


def get_dataframe(run: int) -> pd.DataFrame:
    ps1 = get_data("ps1")[run, :]
    ps2 = get_data("ps2")[run, :]
    ps3 = get_data("ps3")[run, :]
    # ps4 = get_data("ps4")[run, :]  # all values are 0
    ps5 = get_data("ps5")[run, :]
    ps6 = get_data("ps6")[run, :]
    eps1 = get_data("eps1")[run, :]
    data100 = [ps1, ps2, ps3, ps5, ps6, eps1]
    col100 = ["ps1", "ps2", "ps3", "ps5", "ps6", "eps1"]

    fs1 = resize(get_data("fs1")[run, :], 10)
    fs2 = resize(get_data("fs2")[run, :], 10)
    data10 = [fs1, fs2]
    col10 = ["fs1", "fs2"]

    ts1 = resize(get_data("ts1")[run, :], 1)
    ts2 = resize(get_data("ts2")[run, :], 1)
    ts3 = resize(get_data("ts3")[run, :], 1)
    ts4 = resize(get_data("ts4")[run, :], 1)
    vs1 = resize(get_data("vs1")[run, :], 1)
    se = resize(get_data("se")[run, :], 1)
    cp = resize(get_data("cp")[run, :], 1)
    ce = resize(get_data("ce")[run, :], 1)
    data1 = [ts1, ts2, ts3, ts4, vs1, se, cp, ce]
    col1 = ["ts1", "ts2", "ts3", "ts4", "vs1", "se", "cp", "ce"]

    data = data100 + data10 + data1
    col = col100 + col10 + col1
    df = crate_dataframe(data, col)
    return df


def resize2(data, hz, shape=(2205, 6000)) -> np.ndarray:
    ts = 100 // hz

    tmp = np.empty(shape)
    tmp[:] = np.nan
    for i in range(shape[0]):
        for j in range(shape[1]):
            if j % ts == 0:
                tmp[i, j] = data[i, j // ts]
        nans, x = nan_helper(tmp[i])
        tmp[i, nans] = np.interp(x(nans), x(~nans), tmp[i, ~nans])
        tmp[i] = tmp[i] / max(tmp[i])
    return tmp


def get_dataframe2() -> np.ndarray:
    ps1 = resize2(get_data("ps1"), 100)
    ps2 = resize2(get_data("ps2"), 100)
    ps3 = resize2(get_data("ps3"), 100)
    # ps4 = get_data("ps4")[run, :]  # all values are 0
    ps5 = resize2(get_data("ps5"), 100)
    ps6 = resize2(get_data("ps6"), 100)
    eps1 = resize2(get_data("eps1"), 100)
    data100 = [ps1, ps2, ps3, ps5, ps6, eps1]
    col100 = ["ps1", "ps2", "ps3", "ps5", "ps6", "eps1"]

    fs1 = resize2(get_data("fs1"), 10)
    fs2 = resize2(get_data("fs2"), 10)
    data10 = [fs1, fs2]
    col10 = ["fs1", "fs2"]

    ts1 = resize2(get_data("ts1"), 1)
    ts2 = resize2(get_data("ts2"), 1)
    ts3 = resize2(get_data("ts3"), 1)
    ts4 = resize2(get_data("ts4"), 1)
    vs1 = resize2(get_data("vs1"), 1)
    se = resize2(get_data("se"), 1)
    cp = resize2(get_data("cp"), 1)
    ce = resize2(get_data("ce"), 1)
    data1 = [ts1, ts2, ts3, ts4, vs1, se, cp, ce]
    col1 = ["ts1", "ts2", "ts3", "ts4", "vs1", "se", "cp", "ce"]

    data = data100 + data10 + data1
    arr = np.array(data).transpose((1, 0, 2))
    with open("data_parsed/data", "wb") as f:
        pickle.dump(arr, f)
    return arr


def get_profile_df() -> np.ndarray:
    prof = get_data("profile")
    # col = ["cooler condition", "valve condition", "internal pump leakage", "hydraulic accumulator", "stable flag"]
    return prof


def print_corr(run: int):
    df = get_dataframe(run)
    corr = df.corr()
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    # print(corr)
    attributes = list(df.columns)
    scatter_matrix(df[attributes[0:8]], figsize=(100, 100))
    scatter_matrix(df[attributes[8:16]], figsize=(100, 100))

    df.plot(kind="scatter", x="ps1", y="ps2", alpha=0.1)
    plt.show()


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    scalar = MinMaxScaler()
    scalar.fit(df)
    df = pd.DataFrame(scalar.transform(df), columns=list(df.columns))
    return df


def pca():
    data = get_data("data")
    data = data.reshape((2205, 16 * 6000))

    target = get_profile_df()
    # target[:, 0] = target[:, 0] / 100
    # target[:, 1] = target[:, 1] / 100
    # target[:, 2] = target[:, 2] / 2
    # target[:, 3] = target[:, 3] / 130

    scaler = StandardScaler()
    scaler.fit(data)
    train_data_sc = scaler.transform(data)

    componest = 1000

    pca = PCA(n_components=componest)
    train_data_pca = pca.fit_transform(train_data_sc)
    pcaStd = np.std(train_data_pca)

    model = models.Sequential()
    layer = 1
    units = 128

    model.add(layers.Dense(units, input_dim=componest, activation='relu'))
    model.add(layers.GaussianNoise(pcaStd))
    for i in range(layer):
        model.add(layers.Dense(units, activation='relu'))
        model.add(layers.GaussianNoise(pcaStd))
        model.add(layers.Dropout(0.1))
    model.add(layers.Dense(5, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['categorical_accuracy'])

    model.fit(train_data_pca, target, epochs=100, batch_size=256, validation_split=0.15, verbose=2)


def main():
    # pca()

    # get_dataframe2()

    data = get_data("data")
    # print(data)
    target = get_profile_df()
    target[:, 0] = target[:, 0] / 100
    # target[:, 1] = target[:, 1] / 100
    # target[:, 2] = target[:, 2] / 2
    # target[:, 3] = target[:, 3] / 130
    print(target[:, 0])
    print(target.shape)

    data = data.reshape((2205, 16 * 6000))
    print(data.shape)
    print(data)

    network = models.Sequential()
    network.add(layers.Dense(1024, activation="relu", input_shape=(16 * 6000,)))
    # network.add(layers.Dense(512, activation="relu"))
    # network.add(layers.Dropout(0.1))
    network.add(layers.Dense(1, activation="relu"))
    network.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
    network.fit(data, target[:, 0], epochs=5, batch_size=100)


if __name__ == '__main__':
    main()
