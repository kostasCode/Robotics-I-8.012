import matplotlib.pyplot as plt
from matplotlib import animation
from ROBLAB import * 

data = workspace_demo(plot=False)
points = data[:,:3]
joint_values = data[:, 3:]

q, xyz = helper_plot([1, 0, 0],
                     [1, 2, 3],
                     10, points, joint_values, order=5, plot=False)

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
    
    q1, q2, q3 = q[frame]
    plot_robot(q1, q2, q3, axis=ax, plot=True)
    return ax,

ani = animation.FuncAnimation(fig, update, frames=len(q), interval=100, blit=False)
plt.show()

plt.close('all')