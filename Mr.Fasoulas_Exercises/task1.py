import numpy as np 
from rob.csrl_robotics import RobotUtils, RobotViz
 
utils = RobotUtils()
viz = RobotViz()

R = np.eye(4)

F = utils.gtransl(1,4,0.1) @ utils.grotz(np.pi) @ R

O =  F @ utils.gtransl(0.31 , 0.31 , 0) @ utils.grotz(-np.pi/2)

# EH = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,-160],[0,0,0,1]])

gRC1 = utils.gtransl(5,1,0) @ utils.grotz(-np.pi/2) @ utils.grotx((np.pi/2)) @ R

# gCE1 = utils.gtransl(0.2,0.2,2.5) @ utils.grotz(-np.pi) @ R

rc1 = np.array([[0, 0, -1, 500],
                [-1, 0, 0, 100],
                [0, 1, 0, 0],
                [0, 0, 0, 1]])

ce1 = np.array([[1, 0, 0, 20],
                [0, -1, 0, 20],
                [0, 0, -1, 250],
                [0, 0, 0, 1]])

het = np.array([[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, -160],
                [0, 0, 0, 1]])

rc2 = rc1.copy()
ce2 = np.array([[1, 0, 0, 20],
                [0, -1, 0, 20],
                [0, 0, -1, 110],
                [0, 0, 0, 1]])

ce3 = ce2.copy()
ce4 = ce2.copy()

rh3 = np.array([[1, 0, 0, 49],
                [0, 1, 0, 349],
                [0, 0, 1, 110],
                [0, 0, 0, 1]])

rh4 = np.array([[1, 0, 0, 49],
                [0, 1, 0, 349],
                [0, 0, 1, 0],
                [0, 0, 0, 1]])


hr1 = rc1 @ ce1 @ het
hr2 = rc2 @ ce2 @ het
hr3 = rh3 @ ce3 @ het
hr4 = rh4 @ ce4 @ het

print("gRH1:\n", hr1)
print("gRH2:\n", hr2)
print("gRH3:\n", hr3)
print("gRH4:\n", hr4)

# final results scaled to be in frame 
H1 = np.array([[0, 0, 1, 0.9],
               [-1, 0, 0, 0.8],
               [0, -1, 0, 0.2],
               [0, 0, 0, 1]]) 

H2 = np.array([[0, 0, 1, 2.3],
               [-1, 0, 0, 0.8],
               [0, -1, 0, 0.2],
               [0, 0, 0, 1]])

H3 = np.array([[1, 0, 0, 0.69],
               [0, -1, 0, 3.69],
               [0, 0, -1, 3.8],
               [0, 0, 0, 1]])

H4 = np.array([[1, 0, 0, 0.69],
               [0, -1, 0, 3.69],
               [0, 0, -1, 2.7],
               [0, 0, 0, 1]])

viz.add_frame(R, "R")
viz.add_frame(gRC1, "C")
viz.add_frame(F, "F")
viz.add_frame(O, "O")

viz.add_frame(H1, "H1")
viz.add_frame(H2, "H2")
viz.add_frame(H3, "H3")
viz.add_frame(H4, "H4")


# Display all stored plots at once
viz.show_plots()
