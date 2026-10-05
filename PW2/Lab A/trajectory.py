import numpy as np
import matplotlib.pyplot as plt

trajectory = np.loadtxt("trajectory.csv", delimiter = ",", skiprows=1)

t = trajectory[:, 0]
x = trajectory[:, 1]
y = trajectory[:, 2]

#calculating velocity components
vx = np.gradient(x, t)
vy = np.gradient(y, t)

#calculate speed
speed = np.sqrt(vx**2 + vy**2)

#plotting the trajectory
plt.figure()
plt.plot(x, y)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("2D Trajectory")
plt.grid(True)
plt.tight_layout()
plt.savefig("trajectory.png")
plt.close()

#plotting speed over time
plt.figure()
plt.plot(t, speed)
plt.xlabel("Time (s)")
plt.ylabel("Speed (m/s)")
plt.grid(True)
plt.tight_layout()
plt.savefig("speed.png")
plt.close()