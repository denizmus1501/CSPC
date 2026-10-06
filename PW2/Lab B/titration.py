"""
PW2 Lab B Part 5 (bonus) -- find a titration's equivalence point.

titration.csv holds a titration curve: pH versus the volume of base added.
The equivalence point is the volume where the pH changes fastest (the steep
jump). Numerically, that is where the SLOPE of the pH curve is largest.
Run:  python titration.py
"""
import numpy as np 
import matplotlib.pyplot as plt

# TODO 1: read titration.csv (columns volume_base, pH) into arrays V, pH.

data = np.loadtxt("titration.csv", delimiter = ",", skiprows = 1)

V = data[:, 0]
pH = data[:, 1]

# TODO 2: compute the slope of the pH curve with np.gradient(pH, V), and find
#         the volume where that slope is largest (np.argmax). That is the
#         equivalence point. Print it.

slope = np.gradient(pH, V)
eq_point = np.argmax(slope)
print(f"The equivalence point: {eq_point}mL")

# TODO 3: make two plots side by side: (left) pH vs volume with a line at the
#         equivalence point; (right) the slope vs volume, showing it peaks
#         at the equivalence point. Save as titration.png.

fig, axes = plt.subplots(1, 2)

axes[0].plot(V, pH)
axes[0].axvline(eq_point, linestyle="--")
axes[0].set_xlabel("Volume")
axes[0].set_ylabel("pH")
axes[0].set_title("pH vs Volume")

axes[1].plot(V, slope)
axes[1].axvline(eq_point, linestyle="--")
axes[1].set_xlabel("Volume")
axes[1].set_ylabel("Slope")
axes[1].set_title("Slope vs Volume")

plt.tight_layout()
plt.savefig("titration.png")
