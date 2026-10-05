"""
PW2 Lab A -- Motion from tracking data.

Read noisy free-fall position measurements, then:
  - differentiate once  -> velocity
  - differentiate twice -> acceleration (should be ~ constant -g, but noisy!)
  - integrate the acceleration back up -> recover velocity and position

Complete the TODOs. Run:  python analysis.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

# TODO 1: read freefall.csv into arrays t and y
#         (hint: np.loadtxt with a comma delimiter, skipping the header)

data = np.loadtxt("freefall.csv", delimiter = ',', skiprows = 1)

t = data[:, 0]
y = data[:, 1]

# TODO 2: compute velocity v = derivative of y w.r.t. t   (np.gradient)
#         and acceleration a = derivative of v w.r.t. t    (np.gradient again)
#         Print the mean acceleration. Is it close to -9.81? Is it noisy?

v = np.gradient(y, t)
a = np.gradient(v, t)

print(f"Acceleration found from derivation: {np.mean(a):.4f}") #-8.57968750000008
SD = a.std()
print(f"Standard deviation: {SD:.4f}")
SE = SD/(a.size**(1/2))
print(f"Standard error: {SE:.4f}") #shows that our found acceleration is valid

# TODO 3: integrate a back up to recover velocity and position
#         (hint: cumulative_trapezoid(a, t, initial=0) + v[0], then again)

v_recovered = cumulative_trapezoid(a, t, initial = 0) + v[0]
y_recovered = cumulative_trapezoid(v_recovered, t, initial = 0) + y[0]

difference = np.abs(y_recovered - y)
max_difference = np.max(difference)

print(f"Maximum position difference: {max_difference:.4f} m")

# TODO 4: make a figure with 3 stacked panels: position, velocity, acceleration
#         vs time. Mark the true -9.81 line on the acceleration panel.
#         Save it as motion.png

fig, axes = plt.subplots(3, 1, sharex = True)

axes[0].plot(t, y)
axes[0].set_ylabel("Position y (m)")
axes[0].set_title("Free-fall motion from tracking data")
axes[0].grid(True)

axes[1].plot(t, v)
axes[1].set_ylabel("Velocity v (m/s)")
axes[1].grid(True)

axes[2].plot(t, a)
axes[2].set_ylabel("Acceleration a (m/s^2)")
axes[2].grid(True)
axes[2].set_xlabel("Time t (s)")
axes[2].axhline(-9.81, linestyle = "--", label = "True -9.81 m/s^2")
axes[2].legend()

plt.tight_layout()
plt.savefig("motion.png")
plt.close()

#