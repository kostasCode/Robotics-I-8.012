import numpy as np 
from rob.csrl_robotics import RobotUtils, RobotViz
 
utils = RobotUtils()
viz = RobotViz()

theta1 = 30
theta2 = 30

t1x = 2
t1y = 2

t2x = 8
t2y = 1 


theta_rad1 = np.radians(theta1)
theta_rad2 = np.radians(theta2)


gBT = np.array([[0, 0, -1, 3],
                [1, 0, 0, 4],
                [0, -1, 0, 5],
                [0, 0, 0, 1]])


tr1 = utils.gtransl(t1x , t1y ,0)
tr2 = utils.gtransl(t2x , t2y ,0)

gT1 = gBT @ tr1 @ utils.grotz(-theta_rad1)  

gT2 = gBT @ tr2 @ utils.grotz(theta_rad2) @ utils.grotx(-np.pi/2) 

viz.add_frame(gBT, "BT")
viz.add_frame(gT1, "1")
viz.add_frame(gT2, "2")

viz.show_plots()