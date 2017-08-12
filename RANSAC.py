import Constants as const
import CudaConfig as ccfg
import Dm3Reader3 as dm3
import ImageSupport as imsup

from numba import cuda
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import copy

# ------------------------------------------------------------

min_tilt = 0
max_tilt = 360
n_iterations = 1000
min_dist_from_model = 4        # 4 px
n_inliers_threshold = 5000

# ------------------------------------------------------------

class Ellipse:
    def __init__(self, center=(0, 0), a=0, b=0, tau=0.0):
        self.center = center
        self.a = a
        self.b = b
        self.tau = tau
        self.F1 = [0, 0]
        self.F2 = [0, 0]
        self.s = 2 * self.a
        self.update()

    def copy(self):
        return copy.deepcopy(self)

    def update(self):
        c = np.sqrt(np.abs(self.a ** 2 - self.b ** 2))
        self.F1 = [-c * np.cos(self.tau), -c * np.sin(self.tau)]
        self.F2 = [c * np.cos(self.tau), c * np.sin(self.tau)]
        self.s = 2 * self.a

# ------------------------------------------------------------

def deg2rad(deg):
    return np.pi * deg / 180.0

# ------------------------------------------------------------

def rad2deg(rad):
    return 180.0 * rad / np.pi

# ------------------------------------------------------------

# def create_ellipse_from_3pts(A, B, C):
#     ellipse = Ellipse(A, 0, 0)
#     x1, y1 = B
#     x2, y2 = C
#     ellipse.a = np.sqrt(((x1 * y2) ** 2 - (x2 * y1) ** 2) / (y2 ** 2 - y1 ** 2))
#     ellipse.b = y1 * ellipse.a / np.sqrt(ellipse.a ** 2 - x1 ** 2)
#     return ellipse

# ------------------------------------------------------------

# def create_ellipse_from_2pts_and_tilt(ept1, ept2, tau):
#     x1, y1 = ept1
#     x2, y2 = ept2
#     a = np.sqrt(np.abs(((x1 * y2) ** 2 - (x2 * y1) ** 2) / (y2 ** 2 - y1 ** 2)))
#     b = np.abs(y1 * a / np.sqrt(np.abs(a ** 2 - x1 ** 2)))
#     tau = deg2rad(tau)
#     return Ellipse([0, 0], a, b, tau)

# ------------------------------------------------------------

def create_ellipse_from_1pt_and_tilt(pt, tau):
    ptx, pty = pt
    tau = deg2rad(tau)
    pt_ang = np.arctan2(ptx, pty)
    tau_coeff = np.cos(tau) + np.sin(tau) * np.tan(tau)
    a = (ptx + pty * np.tan(tau)) / (np.cos(pt_ang) * tau_coeff)
    b = (pty - ptx * np.tan(tau)) / (np.sin(pt_ang) * tau_coeff)
    return Ellipse([0, 0], a, b, tau)

# ------------------------------------------------------------

def calc_dist_from_ellipse(pt, ellipse):
    ptx, pty = pt
    F1 = ellipse.F1
    F2 = ellipse.F2
    dist = np.sqrt((ptx - F1[0]) ** 2 + (pty - F1[1]) ** 2) + np.sqrt((ptx - F2[0]) ** 2 + (pty - F2[1]) ** 2)
    return dist

# ------------------------------------------------------------

# y = row, x = col
# @cuda.autojit
@cuda.jit('void(float32[:, :], float32[:], float32[:])')
def calc_dists_from_ellipse_dev(dists, F1, F2):
    x, y = cuda.grid(2)
    if x >= dists.shape[1] or y >= dists.shape[0]:
        return

    x0 = x - dists.shape[1] // 2
    y0 = y - dists.shape[0] // 2
    dists[y, x] = math.sqrt((x0 - F1[0]) ** 2 + (y0 - F1[1]) ** 2) + math.sqrt((x0 - F2[0]) ** 2 + (y0 - F2[1]) ** 2)

# ------------------------------------------------------------

def correct_model_ellipse_axis(e, da, first=True):
    new_e = e.copy()
    if first:
        new_e.a = new_e.a + da
    else:
        new_e.b = new_e.b + da
    new_e.update()
    return new_e

# ------------------------------------------------------------

def correct_model_ellipse_tilt(e, dtau):
    new_e = e.copy()
    new_e.tau = new_e.tau + dtau
    new_e.update()
    return new_e

# ------------------------------------------------------------

def correct_model_randomly(e, a_max_len, corr_dir, opts, opt_fixed=False, opt=0):
    if not opt_fixed:
        opt = random.choice(opts)
    step = random.randint(1, 100) * corr_dir
    # TODO: if n_inliers achieve some treshold value (5000), decrease step
    if opt == 0 and e.a + step < a_max_len:
        new_e = correct_model_ellipse_axis(e, step, True)
        # print('Correcting: d_axis1 = {0} px'.format(step))
    elif opt == 1 and e.b + step < a_max_len:
        new_e = correct_model_ellipse_axis(e, step, False)
        # print('Correcting: d_axis2 = {0} px'.format(step))
    else:
        step_rad = deg2rad(step) % (2 * np.pi)
        new_e = correct_model_ellipse_tilt(e, step_rad)
        # print('Correcting: d_tau = {0} deg'.format(step))

    # print('[a = {0}, b = {1}, tilt = {2:.0f}]'.format(new_e.a, new_e.b, rad2deg(new_e.tau % (2 * np.pi))))
    return new_e, opt

# ------------------------------------------------------------

def count_inliers(data, ellipse, disp_ellipse=False):
    min_intensity = np.min(data)
    max_intensity = np.max(data)
    intensity_threshold = 1.0 * (min_intensity + max_intensity) / 2.0

    dists_d = cuda.to_device(np.zeros(data.shape, dtype=np.float32))
    # zero_fix można uwzględnić w F1_d, F2_d zamiast w kernelu
    F1_d = cuda.to_device(np.array(ellipse.F1).astype(np.float32))
    F2_d = cuda.to_device(np.array(ellipse.F2).astype(np.float32))

    blockDim, gridDim = ccfg.DetermineCudaConfigNew(data.shape)
    calc_dists_from_ellipse_dev[gridDim, blockDim](dists_d, F1_d, F2_d)

    dists = dists_d.copy_to_host()
    dists = np.abs(dists - ellipse.s)

    pass_matrix1 = data > intensity_threshold
    pass_matrix2 = dists < min_dist_from_model
    pass_matrix = pass_matrix1 * pass_matrix2
    n_inl = sum(sum(pass_matrix))

    if disp_ellipse and n_inl > n_inliers_threshold:
        display_ellipse_and_neighbour_pixels(ellipse, pass_matrix)

    return n_inl

# ------------------------------------------------------------

def cut_ellipse_from_image(e, img):
    pass

# ------------------------------------------------------------

def display_ellipse_on_image(e, img):
    x = np.zeros(360, dtype=np.float32)
    y = np.zeros(360, dtype=np.float32)
    for t in range(360):
        x[t] = e.a * np.cos(e.tau) * np.cos(deg2rad(t)) - e.b * np.sin(e.tau) * np.sin(deg2rad(t)) + img.shape[1] // 2
        y[t] = e.a * np.sin(e.tau) * np.cos(deg2rad(t)) + e.b * np.cos(e.tau) * np.sin(deg2rad(t)) + img.shape[0] // 2

    plt.imshow(img, cmap='gray')
    plt.plot(x, y, 'r')
    plt.show()

# ------------------------------------------------------------

def display_ellipse_and_neighbour_pixels(e, pass_matrix):
    x = np.zeros(360, dtype=np.float32)
    y = np.zeros(360, dtype=np.float32)
    for t in range(360):
        x[t] = e.a * np.cos(e.tau) * np.cos(deg2rad(t)) - e.b * np.sin(e.tau) * np.sin(deg2rad(t)) + pass_matrix.shape[1] // 2
        y[t] = e.a * np.sin(e.tau) * np.cos(deg2rad(t)) + e.b * np.cos(e.tau) * np.sin(deg2rad(t)) + pass_matrix.shape[0] // 2

    plt.imshow(pass_matrix, cmap='nipy_spectral')
    plt.plot(x, y, 'r')
    plt.show()

# ------------------------------------------------------------

fft1 = dm3.ReadDm3File('ellipse.dm3')
# fft1 = dm3.ReadDm3File('fft1_neg.dm3')

for iteration in range(n_iterations):
    print('--------------------------')
    print('Iteration no {0}...'.format(iteration + 1))
    print('--------------------------')
    tilt_angle_deg = random.randint(min_tilt, max_tilt)
    tilt_angle_rad = deg2rad(tilt_angle_deg)

    a_axis = random.randint(0, fft1.shape[0] // 2)
    b_axis = a_axis     # start from circle
    # b_axis = random.randint(0, fft1.shape[1] // 2)
    model_ellipse = Ellipse([0, 0], a_axis, b_axis, tilt_angle_rad)

    n_inliers_curr = count_inliers(fft1, model_ellipse)
    # print('[a = {0}, b = {1}, tilt = {2:.0f}]'.format(model_ellipse.a, model_ellipse.b, rad2deg(model_ellipse.tau % (2 * np.pi))))
    # print('N0 = {0}'.format(n_inliers_curr))

    # correct model randomly
    n_inliers_dev = n_inliers_curr
    last_opt = 0
    opts = [0, 1, 2]
    all_opts_used = False
    n_tries = 0

    while not all_opts_used or n_inliers_curr > 2000:

        if all_opts_used:   # start again if n_inliers > 2000
            opts = [0, 1, 2]
            all_opts_used = False
            n_tries += 1
            if n_tries > 20:
                break

        if n_inliers_dev <= n_inliers_curr:
            corr_dir = 1
            model_ellipse_dev, last_opt = correct_model_randomly(model_ellipse, fft1.shape[0] // 2, corr_dir, opts)
            opts.remove(last_opt)
        else:
            model_ellipse_dev, last_opt = correct_model_randomly(model_ellipse, fft1.shape[0] // 2, corr_dir, opts,
                                                                 opt_fixed=True, opt=last_opt)

        n_inliers_curr = count_inliers(fft1, model_ellipse)
        n_inliers_dev = count_inliers(fft1, model_ellipse_dev, True)
        print('N1 = {0}, N0 = {1}'.format(n_inliers_dev, n_inliers_curr))

        if n_inliers_dev <= n_inliers_curr and corr_dir > 0:
            corr_dir *= -1
            model_ellipse_dev, last_opt = correct_model_randomly(model_ellipse, fft1.shape[0] // 2, corr_dir, opts,
                                                                 opt_fixed=True, opt=last_opt)
            n_inliers_curr = count_inliers(fft1, model_ellipse)
            n_inliers_dev = count_inliers(fft1, model_ellipse_dev)
            print('N1 = {0}, N0 = {1}'.format(n_inliers_dev, n_inliers_curr))

        if n_inliers_dev > n_inliers_curr:
            # print('Correction accepted!')
            model_ellipse = model_ellipse_dev.copy()
            n_inliers_curr = n_inliers_dev

        if len(opts) == 0:
            all_opts_used = True

        if n_inliers_dev > n_inliers_threshold:
            display_ellipse_on_image(model_ellipse, fft1)
