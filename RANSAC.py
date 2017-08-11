import Constants as const
import Dm3Reader3 as dm3
import ImageSupport as imsup

import numpy as np
from random import randint
import matplotlib.pyplot as plt

# ------------------------------------------------------------

n_model_points = 1
min_tilt = 0
max_tilt = 360
n_iterations = 1000
min_dist_from_model = 5.0        # 5 px
corr_step_default = 10           # 10 px

# ------------------------------------------------------------

class Ellipse:
    def __init__(self, orig=(0, 0), a=0, b=0, tau=0.0):
        self.orig = orig
        self.a = a
        self.b = b
        self.tau = tau
        self.F1 = [0, 0]
        self.F2 = [0, 0]
        self.s = 2 * self.a
        self.update()

    def update(self):
        c = np.sqrt(np.abs(self.a ** 2 - self.b ** 2))
        self.F1 = [-c * np.cos(self.tau), -c * np.sin(self.tau)]
        self.F2 = [c * np.cos(self.tau), c * np.sin(self.tau)]
        self.s = 2 * self.a

# ------------------------------------------------------------

# def create_ellipse_from_3pts(A, B, C):
#     ellipse = Ellipse(A, 0, 0)
#     x1, y1 = B
#     x2, y2 = C
#     ellipse.a = np.sqrt(((x1 * y2) ** 2 - (x2 * y1) ** 2) / (y2 ** 2 - y1 ** 2))
#     ellipse.b = y1 * ellipse.a / np.sqrt(ellipse.a ** 2 - x1 ** 2)
#     return ellipse

# ------------------------------------------------------------

def deg2rad(deg):
    return np.pi * deg / 180.0

# ------------------------------------------------------------

def rad2deg(rad):
    return 180.0 * rad / np.pi

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

def correct_model_ellipse_axis(e, da, first=True):
    if first:
        e.a = e.a + da
    else:
        e.b = e.b + da
    e.update()
    return e

# ------------------------------------------------------------

def correct_model_ellipse_tilt(e, dtau):
    e.tau = e.tau + dtau
    e.update()
    return e

# ------------------------------------------------------------

def count_inliers(data, ellipse):
    min_intensity = np.min(data)
    max_intensity = np.max(data)
    intensity_threshold = 1.0 * (min_intensity + max_intensity) / 2.0
    n_inl = 0

    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            if data[y, x] > intensity_threshold:
                dist_from_model = calc_dist_from_ellipse([y, x], ellipse)
                if np.abs(dist_from_model - ellipse.s) < min_dist_from_model:
                    n_inl += 1

    return n_inl

# ------------------------------------------------------------

def display_ellipse(e):
    x = np.zeros(360, dtype=np.float32)
    y = np.zeros(360, dtype=np.float32)
    for t in range(360):
        x[t] = e.a * np.cos(e.tau) * np.cos(deg2rad(t)) - e.b * np.sin(e.tau) * np.sin(deg2rad(t))
        y[t] = e.a * np.sin(e.tau) * np.cos(deg2rad(t)) + e.b * np.cos(e.tau) * np.sin(deg2rad(t))
    plt.plot(x, y, 'ro')
    plt.show()

# ------------------------------------------------------------

fft1 = dm3.ReadDm3File('fft2.dm3')

# A = [1, 1]
# tilt_angle_deg = 10
# e = create_ellipse_from_1pt_and_tilt(A, tilt_angle_deg)
# x = np.zeros(360, dtype=np.float32)
# y = np.zeros(360, dtype=np.float32)
# for t in range(360):
#     x[t] = e.a * np.cos(e.tau) * np.cos(deg2rad(t)) - e.b * np.sin(e.tau) * np.sin(deg2rad(t))
#     y[t] = e.a * np.sin(e.tau) * np.cos(deg2rad(t)) + e.b * np.cos(e.tau) * np.sin(deg2rad(t))
#
# plt.plot(x, y, 'ro')
# plt.show()

for iteration in range(n_iterations):
    # model_points = []
    # corr_dir = +1

    print('Iteration no {0}...'.format(iteration))
    # for pt_idx in range(n_model_points):
    #     rand_x = randint(0, fft1.shape[1])
    #     rand_y = randint(0, fft1.shape[0])
    #     model_points.append([rand_x, rand_y])
    tilt_angle_deg = randint(min_tilt, max_tilt)
    tilt_angle_rad = deg2rad(tilt_angle_deg)
    # model_points = np.array([ [randint(0, fft1.shape[0]), randint(0, fft1.shape[1])] for i in range(n_model_points) ])

    # create ellipse from 1 point (on ellipse, and origin = [0, 0]) and tilt angle (in degrees)
    # el_pt = model_points[0]
    # model_ellipse = create_ellipse_from_1pt_and_tilt(el_pt, tilt_angle_deg)
    a_axis = randint(0, fft1.shape[0] // 2)
    b_axis = randint(0, fft1.shape[1] // 2)
    model_ellipse = Ellipse([0, 0], a_axis, b_axis, tilt_angle_rad)
    # display_ellipse(model_ellipse)

    n_inliers = count_inliers(fft1, model_ellipse)
    print('Number of inliers: {0}'.format(n_inliers))

    # correct a
    n_inliers_prev = -1
    corr_step = corr_step_default
    while n_inliers > n_inliers_prev and model_ellipse.a < fft1.shape[0] // 2:
        # display_ellipse(model_ellipse)
        model_ellipse = correct_model_ellipse_axis(model_ellipse, corr_step, True)
        n_inliers_prev = n_inliers
        n_inliers = count_inliers(fft1, model_ellipse)
        print('Correcting a: {0}'.format(model_ellipse.a))
        print('Number of inliers: {0}'.format(n_inliers))
        if n_inliers <= n_inliers_prev and corr_step > 0:
            corr_step *= -1
            model_ellipse = correct_model_ellipse_axis(model_ellipse, corr_step, True)
            n_inliers_prev = n_inliers
            n_inliers = count_inliers(fft1, model_ellipse)
            print('Correcting a: {0}'.format(model_ellipse.a))
            print('Number of inliers: {0}'.format(n_inliers))

    # correct b
    n_inliers_prev = -1
    corr_step = corr_step_default
    while n_inliers > n_inliers_prev and model_ellipse.b < fft1.shape[0] // 2:
        # display_ellipse(model_ellipse)
        model_ellipse = correct_model_ellipse_axis(model_ellipse, corr_step, False)
        n_inliers_prev = n_inliers
        n_inliers = count_inliers(fft1, model_ellipse)
        print('Correcting b: {0}'.format(model_ellipse.b))
        print('Number of inliers: {0}'.format(n_inliers))
        if n_inliers <= n_inliers_prev and corr_step > 0:
            corr_step *= -1
            model_ellipse = correct_model_ellipse_axis(model_ellipse, corr_step, False)
            n_inliers_prev = n_inliers
            n_inliers = count_inliers(fft1, model_ellipse)
            print('Correcting b: {0}'.format(model_ellipse.b))
            print('Number of inliers: {0}'.format(n_inliers))

    # correct tilt
    n_inliers_prev = -1
    corr_step = corr_step_default
    while n_inliers > n_inliers_prev:
        # display_ellipse(model_ellipse)
        model_ellipse = correct_model_ellipse_tilt(model_ellipse, deg2rad(corr_step))
        n_inliers_prev = n_inliers
        n_inliers = count_inliers(fft1, model_ellipse)
        print('Correcting tilt: {0:.2f}'.format(rad2deg(model_ellipse.tau)))
        print('Number of inliers: {0}'.format(n_inliers))
        if n_inliers <= n_inliers_prev and corr_step > 0:
            corr_step *= -1
            model_ellipse = correct_model_ellipse_tilt(model_ellipse, deg2rad(corr_step))
            n_inliers_prev = n_inliers
            n_inliers = count_inliers(fft1, model_ellipse)
            print('Correcting tilt: {0:.2f}'.format(rad2deg(model_ellipse.tau)))
            print('Number of inliers: {0}'.format(n_inliers))

    if n_inliers > 2000:
        display_ellipse(model_ellipse)





