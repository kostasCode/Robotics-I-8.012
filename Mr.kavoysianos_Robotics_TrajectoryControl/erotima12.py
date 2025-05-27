import numpy as np
import matplotlib.pyplot as plt

x_A, y_A = 80, 10  # cm
x_B, y_B = -20, 70  # cm

def inverse_kinematics(x, y):
    sl = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return theta, sl

def cubic_polynomial(q0, qf, T):
    a0 = q0
    a1 = 0
    a2 = 3 * (qf - q0) / T**2
    a3 = -2 * (qf - q0) / T**3
    return lambda t: a0 + a1*t + a2*t**2 + a3*t**3

def plot_motion(x_path, y_path, vx_list, vy_list, times):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

    ax1.plot(x_path, y_path, marker='o')
    ax1.set_title("Μονοπάτι Άκρου Βραχίονα")
    ax1.set_xlabel("X (cm)")
    ax1.set_ylabel("Y (cm)")
    ax1.grid(True)
    ax1.axis('equal')

    ax2.plot(times, vx_list, label='ΔX/Δt')
    ax2.plot(times, vy_list, label='ΔY/Δt')
    ax2.set_title("Συστατικά Ταχύτητας Άκρου")
    ax2.set_xlabel("Χρόνος (s)")
    ax2.set_ylabel("Ταχύτητα (cm/s)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

# Ερώτημα 1 και 2
T = 1.0  # sec
theta_A, sl_A = inverse_kinematics(x_A, y_A)
theta_B, sl_B = inverse_kinematics(x_B, y_B)

theta_t = cubic_polynomial(theta_A, theta_B, T)
sl_t = cubic_polynomial(sl_A, sl_B, T)

times = np.arange(0, T + 0.05, 0.05)
x_path, y_path = [], []
vx_list, vy_list = [], []

for i, t in enumerate(times):
    theta = theta_t(t)
    sl = sl_t(t)
    x = sl * np.cos(theta)
    y = sl * np.sin(theta)

    x_path.append(x)
    y_path.append(y)

    if i > 0:
        dt = times[i] - times[i-1]
        dx = x - x_path[i-1]
        dy = y - y_path[i-1]
        vx_list.append(dx / dt)
        vy_list.append(dy / dt)
    else:
        vx_list.append(0)
        vy_list.append(0)

plot_motion(x_path, y_path, vx_list, vy_list, times)
