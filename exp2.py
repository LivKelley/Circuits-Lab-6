#!/usr/bin/env python3
# coding=utf-8
import csv
import numpy as np
import matplotlib.pyplot as plt

divider_configs = ["parallel", "single", "series"]
power_configs = ['10mv','vdd']

Vg = {}
I = {}

def load_config_data(divider_config):
    Vg[divider_config] = {}
    I[divider_config] = {}
    for power_config in power_configs:
        file_name = 'data/exp2_nmos_{!s}{!s}.csv'.format(divider_config, '_vdd' if power_config == 'vdd' else '')
        with open(file_name) as f:
            c = csv.reader(f, delimiter=",")
            next(c)
            
            Vg[divider_config][power_config] = []
            I[divider_config][power_config] = []
            for row in c:
                Vg[divider_config][power_config].append(float(row[0]))
                I[divider_config][power_config].append(float(row[1]))

for divider_config in divider_configs:
    load_config_data(divider_config)

for power_config in power_configs:
    fig = plt.figure()
    ax = plt.subplot(111)
    for divider_config in divider_configs:

        ax.semilogy(Vg[divider_config][power_config],I[divider_config][power_config], label=divider_config)
    plt.legend()
    plt.title("Vds={!s}".format(power_config))
    plt.xlabel("Vg (V)")
    plt.ylabel("Channel Current (A)")
    plt.show()

    fig = plt.figure()
    ax = plt.subplot(111)
    print(np.array(Vg['series'][power_config]))
    I_par_sing = np.divide(np.array(I['parallel'][power_config]),np.array(I['single'][power_config]))
    I_ser_sing = np.divide(np.array(I['series'][power_config]),np.array(I['single'][power_config]))
    ax.plot(Vg['single'][power_config],I_par_sing, label="Parallel vs Single")
    ax.plot(Vg['single'][power_config],I_ser_sing, label="Series vs Single")
    plt.legend()
    plt.show()
