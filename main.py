import matplotlib.pyplot as plt
import numpy as np
from data_parser import get_data
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model


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


def main():
    df = get_dataframe(0)
    df = df.interpolate()

    df = normalize(df)
    print(df)


if __name__ == '__main__':
    main()
