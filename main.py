import matplotlib
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from scipy.fft import rfft, rfftfreq, irfft
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import savgol_filter
from scipy.signal import medfilt
import funcs
import matplotlib.pyplot as pt

params = {'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'large',
          'ytick.labelsize': 'large',
          "figure.autolayout": True,
          "figure.figsize": [35.0, 15.0],
          'text.usetex': True,
          }
matplotlib.use('pgf')
pt.rcParams.update(params)
pt.ioff()

noise_pages = PdfPages("noise.pdf")
noise_data = ['DS0004.CSV', 'DS0005.CSV', 'DS0006.CSV']
for file in noise_data:
    raw_data = (np.transpose(pd.read_csv('csv-sources/' + file, usecols=[0, 1]).values[25:]).tolist())
    for i in range(0, len(raw_data)):
        for j in range(0, len(raw_data[0])):
            raw_data[i][j] = float(raw_data[i][j])
    x_data, y_data = raw_data[0], raw_data[1]
    y_savitzky_golay = savgol_filter(y_data, 1001, 2)
    y_median = medfilt(y_data)
    y_both = savgol_filter(y_median, 1001, 2)
    fig, axes = pt.subplots(4, 1)
    funcs.add_graph(x_data, y_data, "Signal", '$t$, $s$', 'Voltage, $V$', axes[0])
    funcs.add_graph(x_data, y_savitzky_golay, "Savitzky–Golay filtered", '$t$, $s$', 'Voltage, $V$', axes[1])
    funcs.add_graph(x_data, y_median, "Median filtered", '$t$, $s$', 'Voltage, $V$', axes[2])
    funcs.add_graph(x_data, y_both, "Both", '$t$, $s$', 'Voltage, $V$', axes[3])
    noise_pages.savefig(fig)
    pt.cla()
    pt.close(fig)
noise_pages.close()

sum_pages = PdfPages("sum.pdf")
sum_data = ['DS0001.CSV', 'DS0002.CSV', 'DS0003.CSV']
for file in sum_data:
    raw_data = (np.transpose(pd.read_csv('csv-sources/' + file, usecols=[0, 1]).values[25:]).tolist())
    sampling_period = float((pd.read_csv('csv-sources/' + file, usecols=[0, 1]).values[18][1]))
    for i in range(0, len(raw_data)):
        for j in range(0, len(raw_data[0])):
            raw_data[i][j] = float(raw_data[i][j])

    x_data, y_data = raw_data[0], medfilt(raw_data[1])
    y_fourier = rfft(y_data)
    x_freq = rfftfreq(len(y_data), sampling_period)
    y_power = [np.abs(y) / len(y_data) for y in y_fourier]
    fig, axes = pt.subplots(5, 1)
    args = argrelextrema(np.array(y_power), np.greater)[0]
    max_value_args = sorted([i for i in args],
                            key=lambda i: y_power[i],
                            reverse=True)[:2]
    first_fourier = y_fourier[max_value_args[0]]
    second_fourier = y_fourier[max_value_args[1]]
    funcs.add_graph(x_data, y_data, "Signal", '$t$, $s$', 'Voltage, $V$', axes[0])
    funcs.add_graph(x_freq, y_power, "Fourier transform result in frequency field",
                    'Frequency, $Hz$',
                    'Modulus of complex number', axes[1])
    for i in range(len(y_fourier)):
        if i not in max_value_args:
            y_fourier[i] = 0
    y_fourier[max_value_args[0]] = 0
    y_first = irfft(y_fourier, len(x_data))
    y_fourier[max_value_args[0]] = first_fourier
    y_fourier[max_value_args[1]] = 0
    y_second = irfft(y_fourier, len(x_data))
    funcs.add_graph(x_data, y_first, "Signal 1", '$t$, $s$', 'Voltage, $V$', axes[2])
    funcs.add_graph(x_data, y_second, "Signal 2", '$t$, $s$', 'Voltage, $V$', axes[3])
    funcs.add_graph(x_data, [y_first[i] + y_second[i] for i in range(len(x_data))], "Signal sum", '$t$, $s$',
                    'Voltage, $V$',
                    axes[4])
    sum_pages.savefig(fig)
    pt.cla()
    pt.close(fig)
sum_pages.close()
