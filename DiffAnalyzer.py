import Constants as const
import Dm3Reader3 as dm3
import ImageSupport as imsup

import numpy as np
from scipy.optimize import curve_fit
from scipy import signal
import matplotlib.pyplot as plt

# ------------------------------------------------------------

def degrees_to_radians(deg):
    return np.pi * deg / 180.0

# ------------------------------------------------------------

def rot_average(data):
    r_max = data.shape[0] // 2
    rot_avg_int = np.zeros(r_max, dtype=np.float32)
    for r in range(1, r_max):
        for theta in np.arange(0., 360., 1., dtype=np.float32):
            theta_rad = degrees_to_radians(theta)
            x = r * np.cos(theta_rad) + r_max
            y = r * np.sin(theta_rad) + r_max
            rot_avg_int[r] += data[y, x]
    return rot_avg_int

# ------------------------------------------------------------

def remove_linear_background(x, y):
    a_coeff = (y[-1] - y[0]) / (x[-1] - x[0])
    b_coeff = y[0] - a_coeff * x[0]
    linear_bkg = x * a_coeff + b_coeff
    return y - linear_bkg

# ------------------------------------------------------------

def gauss(x, *pars):
    A, mean, sigma = pars
    return A * np.exp(-(x-mean)**2/(2.*(sigma**2)))

# ------------------------------------------------------------

# def fit_gauss(x, y):
#     coeff, var_matrix = curve_fit(gauss, x, y, p0=pars0)
#     return coeff, gauss(x, *coeff)

# ------------------------------------------------------------

fft1 = dm3.ReadDm3File('fft1.dm3')
fft2 = dm3.ReadDm3File('fft2.dm3')
fft3 = dm3.ReadDm3File('fft3.dm3')
fft4 = dm3.ReadDm3File('fft4.dm3')

rot_average_int = rot_average(fft1)
r_range = np.arange(0, fft1.shape[0] // 2, 1, dtype=np.float32)

# subtract background
x = r_range[15:]
y = rot_average_int[15:]
# y_no_bkg = signal.detrend(y, type='linear')
y_no_bkg = remove_linear_background(x, y)
plt.plot(x, y_no_bkg, 'ro')
plt.plot(x, y, 'b+')
plt.show()

y = np.copy(y_no_bkg[:])

# fitting gauss

n_peaks = 4
peaks = []
for n in range(n_peaks):
    pars0 = [ np.max(y_no_bkg), y_no_bkg.argmax(axis=0) + x[0], x.shape[0] / 6 ]
    coeff, var_matrix = curve_fit(gauss, x, y_no_bkg, p0=pars0)
    peaks.append(coeff)

    plt.plot(x, y_no_bkg, 'b+', label='data')
    plt.plot(x, gauss(x, *coeff), 'ro:', label='fit')
    plt.show()

    cut_point = int(coeff[1] + 2 * coeff[2]) - x[0]

    x = x[cut_point:]
    y_no_bkg = y_no_bkg[cut_point:]
    plt.plot(x, y_no_bkg, 'b+', label='data')
    plt.show()

for idx, peak in zip(range(n_peaks), peaks):
    peak[1:2] *= const.pxWidth
    print('{0}: {1:.2f} 1/nm'.format(idx+1, peak[1]))