import Constants as const
import CudaConfig as ccfg
import Dm3Reader3 as dm3

from numba import cuda
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import copy

# ------------------------------------------------------------

min_tilt = 0
max_tilt = 360
n_iterations = 200
min_dist_from_model = 5       # 5 px
intensity_coeff = 1.0
try_again_threshold = 4000
n_inliers_threshold = 7000
min_ab_ratio = 0.3
max_n_tries = 20

# ------------------------------------------------------------

class Ellipse:
    def __init__(self, center=(0, 0), a=0, b=0, tau=0.0):
        self.center = center
        self.a = a
        self.b = b
        self.tau = tau
        self.F1 = [0, 0]
        self.F2 = [0, 0]
        self.s = 2 * max(self.a, self.b)
        self.update()

    def copy(self):
        return copy.deepcopy(self)

    def update(self):
        c = np.sqrt(np.abs(self.a ** 2 - self.b ** 2))
        self.F1 = [-c * np.cos(self.tau), -c * np.sin(self.tau)]
        self.F2 = [c * np.cos(self.tau), c * np.sin(self.tau)]
        self.s = 2 * max(self.a, self.b)

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

def correct_model_ellipse_axes(e, da, first=True, second=False):
    new_e = e.copy()
    if first:
        new_e.a = new_e.a + da
    if second:
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

def determine_max_rand_step(n_inl, init_val=300, min_val=10):
    if n_inl <= n_inliers_threshold:
        max_rand_step = init_val
    elif n_inliers_threshold < n_inl <= 2 * n_inliers_threshold:
        a = (min_val - init_val) / n_inliers_threshold
        b = init_val - a * n_inliers_threshold
        max_rand_step = a * n_inl + b
    else:
        max_rand_step = min_val
    return int(max_rand_step)

# ------------------------------------------------------------

def correct_model_randomly(e, a_max_len, corr_dir, n_inl, opts, opt_fixed=False, opt=0):
    if not opt_fixed:
        opt = random.choice(opts)

    max_rand_step = determine_max_rand_step(n_inl, 300, 10)
    max_rand_tilt = determine_max_rand_step(n_inl, 90, 10)

    step = random.randint(1, max_rand_step) * corr_dir
    tilt = random.randint(1, max_rand_tilt) * corr_dir
    a_ok = 0 < e.a + step < a_max_len
    b_ok = 0 < e.b + step < a_max_len

    # --- Method no 1 ---
    # corr_a, corr_b, corr_tau = False
    # if opt in [0, 2, 3] and 0 < e.a + step < a_max_len:
    #     corr_a = True
    # if opt in [1, 2, 3] and 0 < e.b + step < a_max_len:
    #     corr_b = True
    # if opt in [3, 4]:
    #     corr_tau = True

    # --- Method no 2 ---
    corr_a = True if opt in [0, 2, 3] and a_ok else False
    corr_b = True if opt in [1, 2, 3] and b_ok else False
    corr_tau = True if opt in [3, 4] else False

    new_e = correct_model_ellipse_axes(e, step, corr_a, corr_b)
    if corr_tau:
        tilt_rad = deg2rad(tilt) % (2 * np.pi)
        new_e = correct_model_ellipse_tilt(new_e, tilt_rad)

    # --- Method no 3 ---
    # if opt == 0 and a_ok:
    #     new_e = correct_model_ellipse_axes(e, step, corr_a, not corr_b)
    # elif opt == 1 and b_ok:
    #     new_e = correct_model_ellipse_axes(e, step, not corr_a, corr_b)
    # elif opt == 2 and a_ok and b_ok:
    #     new_e = correct_model_ellipse_axes(e, step, corr_a, corr_b)
    # elif opt == 3:
    #     if not a_ok: corr_a = False
    #     if not b_ok: corr_b = False
    #     new_e = correct_model_ellipse_axes(e, step, corr_a, corr_b)
    #     tilt_rad = deg2rad(tilt) % (2 * np.pi)
    #     new_e = correct_model_ellipse_tilt(new_e, tilt_rad)
    # else:
    #     tilt_rad = deg2rad(tilt) % (2 * np.pi)
    #     new_e = correct_model_ellipse_tilt(e, tilt_rad)

    # print('[a = {0}, b = {1}, tilt = {2:.0f}]'.format(new_e.a, new_e.b, rad2deg(new_e.tau % (2 * np.pi))))
    return new_e, opt

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

def calc_dists_from_ellipse(data, ellipse):
    dists_d = cuda.to_device(np.zeros(data.shape, dtype=np.float32))
    # zero_fix można uwzględnić w F1_d, F2_d zamiast w kernelu
    F1_d = cuda.to_device(np.array(ellipse.F1).astype(np.float32))
    F2_d = cuda.to_device(np.array(ellipse.F2).astype(np.float32))

    blockDim, gridDim = ccfg.DetermineCudaConfigNew(data.shape)
    calc_dists_from_ellipse_dev[gridDim, blockDim](dists_d, F1_d, F2_d)

    dists = dists_d.copy_to_host()
    dists = np.abs(dists - ellipse.s)

    return dists

# ------------------------------------------------------------

def count_inliers(data, ellipse, disp_ellipse=False):
    min_intensity = np.min(data)
    max_intensity = np.max(data)
    intensity_threshold = intensity_coeff * (min_intensity + max_intensity) / 2.0

    dists = calc_dists_from_ellipse(data, ellipse)

    pass_matrix1 = data > intensity_threshold
    pass_matrix2 = dists < min_dist_from_model
    pass_matrix = pass_matrix1 * pass_matrix2
    n_inl = sum(sum(pass_matrix))

    if disp_ellipse and n_inl > n_inliers_threshold:
        display_ellipse_and_neighbour_pixels(ellipse, pass_matrix)

    return n_inl

# ------------------------------------------------------------

def cut_ellipse_from_image(ellipse, data, min_dist):
    dists = calc_dists_from_ellipse(data, ellipse)
    pixels_to_keep = dists >= min_dist
    data_new = data * pixels_to_keep.astype(np.int32)
    return data_new

# ------------------------------------------------------------

def frange(x, y, jump):
    while x < y:
        yield x
        x += jump

# ------------------------------------------------------------

def generate_ellipse_to_draw(e, n_points=360):
    x = np.zeros(n_points, dtype=np.float32)
    y = np.zeros(n_points, dtype=np.float32)

    a = max(e.a, e.b)
    b = min(e.a, e.b)
    t_step = max_tilt / n_points

    for t_idx, t in zip(range(n_points), frange(min_tilt, max_tilt, t_step)):
        x[t_idx] = a * np.cos(e.tau) * np.cos(deg2rad(t)) - b * np.sin(e.tau) * np.sin(deg2rad(t))
        y[t_idx] = a * np.sin(e.tau) * np.cos(deg2rad(t)) + b * np.cos(e.tau) * np.sin(deg2rad(t))

    return x, y

# ------------------------------------------------------------

def display_ellipse_on_image(e, img):
    x, y = generate_ellipse_to_draw(e)
    x += img.shape[1] // 2
    y += img.shape[0] // 2

    plt.imshow(img, cmap='gray')
    plt.plot(x, y, 'r')
    plt.show()

# ------------------------------------------------------------

def display_ellipse_and_neighbour_pixels(e, pass_matrix):
    x, y = generate_ellipse_to_draw(e)
    x += pass_matrix.shape[1] // 2
    y += pass_matrix.shape[0] // 2

    plt.imshow(pass_matrix, cmap='nipy_spectral')
    plt.plot(x, y, 'r')
    plt.show()

# ------------------------------------------------------------

# for ang in frange(0.0, 360.0, 45.0):
#     e = Ellipse([0, 0], 100, 200, deg2rad(ang))
#     x, y = generate_ellipse_to_draw(e)
#     plt.plot(x, y, 'r')
# plt.show()

fft1 = dm3.ReadDm3File('ellipse2.dm3')
# fft1 = dm3.ReadDm3File('fft1_mask.dm3')
fitted_ellipses = []
ellipse_to_track = []

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
    if iteration == 0:
        ellipse_to_track.append(model_ellipse)

    n_inliers_curr = count_inliers(fft1, model_ellipse)
    print('New model:')
    print('[a = {0}, b = {1}, tilt = {2:.0f}]'.format(model_ellipse.a, model_ellipse.b, rad2deg(model_ellipse.tau % (2 * np.pi))))
    # print('N0 = {0}'.format(n_inliers_curr))

    # correct model randomly
    n_inliers_dev = n_inliers_curr
    corr_dir = 1
    last_opt = 0
    opts = list(range(0, 5))
    all_opts_used = False
    n_tries = 0

    while not all_opts_used or n_inliers_curr > try_again_threshold:

        if all_opts_used:   # start again if n_inliers > try_again_threshold
            opts = list(range(0, 5))
            all_opts_used = False
            n_tries += 1
            if n_tries > max_n_tries:
                break

        # If new model is worse than the previous one, get a new random correction.
        # If new model is better than the previous one, keep changing the model in the same way
        # (same type of correction, in the same direction).
        if n_inliers_dev <= n_inliers_curr:
            corr_dir = 1
            model_ellipse_dev, last_opt = correct_model_randomly(model_ellipse, fft1.shape[0] // 2, corr_dir, n_inliers_curr, opts)
            opts.remove(last_opt)
        else:
            model_ellipse_dev, last_opt = correct_model_randomly(model_ellipse, fft1.shape[0] // 2, corr_dir, n_inliers_dev, opts,
                                                                 opt_fixed=True, opt=last_opt)

        n_inliers_curr = count_inliers(fft1, model_ellipse)
        n_inliers_dev = count_inliers(fft1, model_ellipse_dev)
        print('N1 = {0}, N0 = {1}'.format(n_inliers_dev, n_inliers_curr))

        # If new model is worse than the previous one (and the direction +1 was already used),
        # change the direction (to -1) and try with the same type of correction.
        if n_inliers_dev <= n_inliers_curr: # and corr_dir > 0:
            corr_dir *= -1
            model_ellipse_dev, last_opt = correct_model_randomly(model_ellipse, fft1.shape[0] // 2, corr_dir, n_inliers_curr, opts,
                                                                 opt_fixed=True, opt=last_opt)

            n_inliers_dev = count_inliers(fft1, model_ellipse_dev)
            print('N1 = {0}, N0 = {1}'.format(n_inliers_dev, n_inliers_curr))

        # If one of applied corrections worked, accept the development model as the new current model.
        if n_inliers_dev > n_inliers_curr:
            # print('Correction accepted!')
            model_ellipse = model_ellipse_dev.copy()
            if iteration == 0:
                ellipse_to_track.append(model_ellipse)

        # Check if all 3 types of corrections were tested against model.
        if len(opts) == 0:
            all_opts_used = True

        # If the current number of inliers is greater than some threshold value,
        # cut the fitted ellipse (and its surrounding) from image, break from
        # current while loop and go to the next iteration.
        # TODO: If ellipse is fitted correctly then next a and b values should be generated from different range,
        # TODO: i.e. there should be a ring of restricted values around a0 and b0 of fitted ellipse.
        # TODO: This should allow for finding the smallest ellipse in 'ellipse2.dm3'.
        if n_inliers_curr > n_inliers_threshold:
            ab_ratio = model_ellipse.a / model_ellipse.b
            if ab_ratio < min_ab_ratio or ab_ratio > (1.0 / min_ab_ratio):
                break
            display_ellipse_on_image(model_ellipse, fft1)
            fft1 = cut_ellipse_from_image(model_ellipse, fft1, 20)
            display_ellipse_on_image(model_ellipse, fft1)
            fitted_ellipses.append(model_ellipse)
            break

    if len(fitted_ellipses) >= 4:
        break

for e, idx in zip(fitted_ellipses, range(len(fitted_ellipses))):
    print('\n-------------\nRing no {0}\n-------------'.format(idx+1))
    print('a = {0:.2f} 1/nm\nb = {1:.2f} 1/nm\ntilt = {2:.0f} deg'.format(e.a * const.pxWidth,
                                                                          e.b * const.pxWidth,
                                                                          rad2deg(e.tau % (2 * np.pi))))

for e in ellipse_to_track:
    x, y = generate_ellipse_to_draw(e)
    plt.plot(x, y, 'r')

plt.show()



