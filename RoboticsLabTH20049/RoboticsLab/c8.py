import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from ROBLAB import *


data = workspace_demo(plot=False)
points = data[:,:3]
joint_values = data[:, 3:]

t = np.linspace(0, 2 * np.pi, 50)

q1_vals = 1.0 + 0.2 * np.sin(t)

q2_vals = np.radians(180) * np.sin(t)

q3_vals = np.radians(45) * np.sin(2*t)

q_all = np.stack([q1_vals, q2_vals, q3_vals], axis=1)

xyz_desired = []

fig_dummy = plt.figure()
ax_dummy = fig_dummy.add_subplot(111, projection='3d')
plt.close(fig_dummy)

for q in q_all:
    p = plot_robot(q[0], q[1], q[2], axis=ax_dummy, plot=False)
    xyz_desired.append(p)

q_from_ikine = []
xyz_actual = []
errors = []

for p in xyz_desired:
    try:
        q_inv = ikine(p, points, joint_values)
        q_from_ikine.append(q_inv)
        p_reconstructed = plot_robot(q_inv[0],q_inv[1],q_inv[2], axis=ax_dummy ,plot=False)
        xyz_actual.append(p_reconstructed)
        err = np.linalg.norm(np.array(p) - np.array(p_reconstructed))
        errors.append(err)
    except Exception as e:
        print(f"Σφάλμα στην ikine για p = {p}: {e}")
        continue

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_zlim([0, 4])
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("PRR Robot Animation")

def update(frame):
    ax.cla()  
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([0, 4])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("PRR Robot Animation")
    
    q1, q2, q3 = q_all[frame]
    plot_robot(q1, q2, q3, axis=ax, plot=True)
    return ax,

ani = animation.FuncAnimation(fig, update, frames=len(q_all), interval=10, blit=False)
plt.show()

fig2 = plt.figure()
plt.plot(errors)
plt.xlabel("Frame")
plt.ylabel("IK Error (Euclidean norm)")
plt.title("Inverse Kinematics Error Over Time")
plt.grid()
plt.show()