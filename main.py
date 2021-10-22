import numpy
import matplotlib.pyplot as plt
from data_parser import get_data


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


def main():
    plot_pressure_data(0)
    plot_flow_data(0)
    plot_temperature_data(0)
    plot_efficiency_data(0)
    plot_vibration_data(0)
    plot_power_data(0)
    plt.show()


if __name__ == '__main__':
    main()
