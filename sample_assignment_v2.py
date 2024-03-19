#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

# Define some parameters
T1 = 0.800  # s
T2star = 0.020  # s
B0 = 1.5  # T
gamma = 2 * np.pi * 42.6 * 1e6  # rad/s/T
w0 = gamma * B0  # rad/s
M0 = np.array([0, 0, 1])  # Magnetisation vector
dt = 0.001  # s
a = np.pi / 2 # rad
TR = T1  # s

t = np.arange(0, TR, dt)  # s, time of the experiment
M_t = np.zeros(shape=(3,) + t.shape)
M_t[:, 0] = M0

# Flip the magnetisation onto transverse plane by an angle a. In this case,
# assuming the B1 field is applied along the y axis,
Rflip = [[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]
M = np.matmul(Rflip, M0)
M_t[:, 1] = M

# Evolution
# magnetisation returns to equilibrium via spin relaxation over time t,
E1 = np.exp(-dt / T1)
E2 = np.exp(-dt / T2star)
E = [[E2, 0, 0], [0, E2, 0], [0, 0, E1]]

# precession is described as a rotation around z
phi = float(w0) * dt

# Create two matrices to describe the combined effect of relaxation and precession
A = np.matmul(
    E, [[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]]
)
B = [0, 0, 1 - E1]
for n in range(2, len(t)):
    M_t[:, n] = np.matmul(A, M_t[:, n - 1]) + B

S = np.sqrt(M_t[0] ** 2 + M_t[1] ** 2)
plt.plot(t, S, "b")
plt.xlabel("time (s)")
plt.ylabel("S")
plt.title("FID")

plt.savefig('sample_assignment_fig.png')
