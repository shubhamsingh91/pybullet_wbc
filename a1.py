#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 17:48:59 2022
pybulet simulation
@author: shubham
1. Testing position control here for a required pose
"""

###############################
# Import the necessary modules
###############################

# The PyBullet physics simulation library
import pybullet as p
import pybullet_data
import os
import time

# Numpy for numerical calculations and manipulations
import numpy as np
import math

# Matplotlib to create the necessary plots
import matplotlib.pyplot as plt

# Functions 
def reset_robot(p,robot):
    for i in range(8):
        p.resetJointState(robot,i,0)

def get_joint_state(p,robot,joint):
    temp1 =p.getJointState(robot,joint)   
    return temp1[0],temp1[1], temp1[2:6]     

def get_all_joint_states(p,robot):
    qvecout=[0]*7
    qdvecout=[0]*7
    jointReactionForces=[0]*6
    tau = [0]*7
    for i in range(7):
        qvecout[i], qdvecout[i],jointReactionForces = get_joint_state(p, robot, i)
    return qvecout,qdvecout    

def position_control(p,robot,targetpos_vec):
    joint_vec = [0,1,2,3,4,5,6]
    force_vec = [123,123,64,64,39,39,39]
    p.setJointMotorControlArray(robot,joint_vec,p.POSITION_CONTROL,targetPositions=targetpos_vec,forces=force_vec)

def velocity_control(p,robot,targetvel):
    for i in range(8):
        p.setJointMotorControl2(robot,i,p.VELOCITY_CONTROL,targetVelocity=targetvel,force=50)

##############################################################
# Create an instance of the Physics Server and connect to it
##############################################################

# Use p.DIRECT to connect to the server without rendering a GUI
# Use p.GUI to create a GUI to render the simulation
#client = p.connect(p.DIRECT) # or p.GUI
client1 = p.DIRECT
client = p.GUI # or p.GUI
p.connect(client)
p.connect(client1)

# Load the URDF of the plane that forms the ground
p.setAdditionalSearchPath(pybullet_data.getDataPath()) # Set the search path to find the plane.urdf file
plane = p.loadURDF("plane.urdf")

# robot import URDF
#str1 = os.getcwd() + "/A02L-M.urdf"
str1 = os.getcwd() + "/model/a1.urdf"
robot = p.loadURDF(str1,flags=p.URDF_USE_SELF_COLLISION,useFixedBase=1)

##################################################
# Set the necessary parameters for the simulation
##################################################

# Set the Gravity vector
p.setGravity(0,0,-9.81, physicsClientId = client)

# Set the simulation time-step
#p.setTimeStep(0.001) #The lower this is, more accurate the simulation 

# Setting all the joints for velocity control    
p.setJointMotorControl2(robot, 0, p.VELOCITY_CONTROL, force=0)
p.setJointMotorControl2(robot, 1, p.VELOCITY_CONTROL, force=0)
p.setJointMotorControl2(robot, 2, p.VELOCITY_CONTROL, force=0)
p.setJointMotorControl2(robot, 3, p.VELOCITY_CONTROL, force=0)
p.setJointMotorControl2(robot, 4, p.VELOCITY_CONTROL, force=0)
p.setJointMotorControl2(robot, 5, p.VELOCITY_CONTROL, force=0)
p.setJointMotorControl2(robot, 6, p.VELOCITY_CONTROL, force=0)
p.setJointMotorControl2(robot, 7, p.VELOCITY_CONTROL, force=0)

print(p.getNumJoints(robot))
print(p.getBasePositionAndOrientation(robot))

timeStep =1/240 # 
p.setTimeStep(timeStep)

# Reset robot
reset_robot(p,robot)

# pos_vec_des = [0.5,0.5,0.5,0.5,0.5,0.5,0.5]

# plotting states
# Initialize lists to store joint states and time
nSteps = 10000


for i in range (nSteps): 
    # position_control(p,robot,pos_vec_des)
    p.stepSimulation()
    # q,qd = get_all_joint_states(p,robot)
    # q_history[i][:]=q
    # qd_history[i][:]=qd
    # time_history[i] = i*timeStep
    time.sleep(timeStep)









