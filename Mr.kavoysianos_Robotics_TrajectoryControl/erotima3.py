import numpy as np
import matplotlib.pyplot as plt

A = np.array([0.8, 0.1])  
B = np.array([-0.2, 0.7])  

V_max = 3.0  # m/s
a_max = 4.0  # m/s^2

D = np.linalg.norm(B - A)

T_acc = V_max / a_max
D_acc = 0.5 * a_max * T_acc**2

if 2 * D_acc > D:
    T_acc = np.sqrt(D / a_max)
    T_total = 2 * T_acc
else:
    D_const = D - 2 * D_acc
    T_const = D_const / V_max
    T_total = 2 * T_acc + T_const

times = np.linspace(0, T_total, 20)
positions = []
dx_dt = []
dy_dt = []
speeds = []

for i, t in enumerate(times):
    if t < T_acc:
        s = 0.5 * a_max * t**2
        v = a_max * t
    elif t < T_total - T_acc:
        s = D_acc + V_max * (t - T_acc)
        v = V_max
    else:
        dt = t - (T_total - T_acc)
        s = D - 0.5 * a_max * (T_acc - dt)**2
        v = a_max * (T_acc - dt)
        
    pos = A + (B - A) * (s / D)
    
    positions.append(pos)
    direction = (B - A) / D
    
    dx_dt.append(direction[0] * v)
    dy_dt.append(direction[1] * v)
    
    speeds.append(v)

positions = np.array(positions)


plt.figure()
plt.plot(positions[:, 0], positions[:, 1], 'o-', label='Πορεία')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Πορεία του άκρου του βραχίονα')
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(times, dx_dt, 'r-', label='ΔΧ/Δt (m/s)')
plt.plot(times, dy_dt, 'b-', label='ΔΥ/Δt (m/s)')
plt.plot(times, speeds, 'g--', label='Ταχύτητα (m/s)')
plt.xlabel('Χρόνος (s)')
plt.ylabel('Ταχύτητα')
plt.title('Προφίλ Ταχύτητας')
plt.legend()
plt.grid(True)
plt.show()

for i, t in enumerate(times):
    print(f"t={t:.2f}s, X={positions[i,0]:.2f}m, Y={positions[i,1]:.2f}m, ΔΧ/Δt={dx_dt[i]:.2f}m/s, ΔΥ/Δt={dy_dt[i]:.2f}m/s")