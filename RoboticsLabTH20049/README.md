# Project Overview

The project is organized into three parts (A, B, C), following a bottom-up approach from geometric vector operations to robot kinematics and motion planning. The simulated mechanism is a PRR (Prismatic-Revolute-Revolute) robotic arm.

## PART A – Vector Geometry & Frame Visualization

- `plot_free_vec(x, c)`, `plot_vec(a, x, c)` – Vector visualization  
- `make_unit(x)` – Normalize a vector  
- `project_vec(a, b)` – Projection of one vector onto another  
- `cross_demo` – 3D cross product visualization  
- `plot_Rot(R)`, `generate_rot()` – Local frames in 3D  

## PART B – Homogeneous Transformations & Applications

- `gRX(th)`, `gRY(th)`, `gRZ(th)` – Return rotation matrix for angle `th` (in degrees) around X, Y, Z axis respectively  
- `gr(R)`, `gp(p)` – Create homogeneous transformation matrices from rotation `R` or translation vector `p`  
- `homogen(R, p)` – Construct full homogeneous transformation matrix from rotation and translation  
- `rotAndTranVec(G, vin)` – Apply homogeneous transformation `G` to a 3D input vector `vin`  
- `plot_hom(G)` – Visualize the local coordinate frame using colored 3D arrows  
- `rotX(th)`, `rotY(th)`, `rotZ(th)` – Return rotation matrix for angle `th` (in degrees) around X, Y, Z axis respectively  
- `rotAndTrans_shape()` – Apply full transformation to 3D geometry (e.g., a cylinder)  

## PART C – PRR Robot and Kinematics

- `g0e(q1, q2, q3)` – Forward kinematics  
- `ikine(p, ...)` – Inverse kinematics via brute-force search  
- `get_trajectory()` – Quintic interpolation of position  
- Full 3D animated demos of end-effector trajectories  


## Joint Space Trajectory - [C6](https://github.com/kostasCode/Robotics-I-8.012/blob/main/RoboticsLabTH20049/RoboticsLab/c6.py)

![ezgif-6eac292a4b2e7a](https://github.com/user-attachments/assets/be6010bd-38e4-41e7-9f52-17c812dc9419)

## Periodic Sinusoidal Trajectory - [C8](https://github.com/kostasCode/Robotics-I-8.012/blob/main/RoboticsLabTH20049/RoboticsLab/c8.py) Dance demo

![ezgif-6da2093036177b](https://github.com/user-attachments/assets/11fad0e8-a84b-468a-984f-7eed75ec4e1f)
