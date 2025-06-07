import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from ROBLAB import * 


q, xyz = helper_plot([1, np.radians(0), np.radians(0)],
                     [1, np.radians(0), np.radians(90)],
                     6.0, points=None, joint_values=None, order=5, space="work", plot=False)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_zlim([0, 4])
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("PRR Robot Animation")

# We'll clear and redraw robot in each frame 
def update(frame):
    ax.cla() 
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([0, 4])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("PRR Robot Animation")
    
    q1, q2, q3 = q[frame]
    plot_robot(q1, q2, q3, axis=ax,)
    return ax

ani = animation.FuncAnimation(fig, update, frames=len(q), interval=100, blit=False)
plt.show()
