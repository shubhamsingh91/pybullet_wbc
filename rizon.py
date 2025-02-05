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
import pinocchio as pin
# import eigenpy
# import boost

# from pybullet_planning import connect, disconnect, load_pybullet, plan_joint_motion, set_joint_positions, LockRenderer

# Numpy for numerical calculations and manipulations
import numpy as np
import math

# Matplotlib to create the necessary plots
import matplotlib.pyplot as plt

# Functions 

def plot_trq(t, torque_commands):
    """
    Plots the torque commands for all joints over time.

    Parameters:
    - t: Time vector.
    - torque_commands: 2D numpy array of shape (n_steps, n_joints), where each row represents the torque command at a time step.
    """
    n_steps, n_joints = torque_commands.shape
    fig, axs = plt.subplots(n_joints, 1, figsize=(10, 2 * n_joints), sharex=True)

    for joint in range(n_joints):
        axs[joint].plot(t, torque_commands[:, joint], label=f'Joint {joint + 1}')
        axs[joint].set_ylabel('Torque [Nm]')
        axs[joint].legend()
        axs[joint].grid(True)

    axs[-1].set_xlabel('Time [s]')
    fig.suptitle('Torque Commands for All Joints')
    plt.tight_layout(rect=[0, 0, 1, 0.96])


def plot_3d_positions(t, positions):
    """
    Plots the 3D positions (x, y, z) over time.

    Parameters:
    - t: time vector
    - positions: 2D numpy array of shape (n, 3), where each row is [x, y, z] position at a time step
    - title: title of the plot
    """
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))

    axs[0].plot(t, positions[:, 0], label='x')
    axs[0].set_ylabel('X Position [m]')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(t, positions[:, 1], label='y')
    axs[1].set_ylabel('Y Position [m]')
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(t, positions[:, 2], label='z')
    axs[2].set_ylabel('Z Position [m]')
    axs[2].set_xlabel('Time [s]')
    axs[2].legend()
    axs[2].grid(True)

    fig.suptitle("ee pose")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
def plot_3d_error(t, positions):
    """
    Plots the 3D positions (x, y, z) over time.

    Parameters:
    - t: time vector
    - positions: 2D numpy array of shape (n, 3), where each row is [x, y, z] position at a time step
    - title: title of the plot
    """
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))

    axs[0].plot(t, positions[:, 0], label='x')
    axs[0].set_ylabel('X error [m]')
    axs[0].set_yscale('log')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(t, positions[:, 1], label='y')
    axs[1].set_ylabel('Y error [m]')
    axs[1].set_yscale('log')
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(t, positions[:, 2], label='z')
    axs[2].set_ylabel('Z error [m]')
    axs[2].set_xlabel('Time [s]')
    axs[2].set_yscale('log')
    axs[2].legend()
    axs[2].grid(True)

    fig.suptitle("ee error")
    plt.tight_layout(rect=[0, 0, 1, 0.96])


def reset_robot(p,robot):
    for i in range(7):
        p.resetJointState(robot,i,0)

def reset_home(p,robot):
    qvec = [0, -40, 0, 90, 0, 40, 0]
    for i in range(p.getNumJoints(robot)-1):
        p.resetJointState(robot,i,np.deg2rad(qvec[i]))

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

def position_control(p,robot,targetpos_vec,posGain):
    joint_vec = [0,1,2,3,4,5,6]
    force_vec = [123,123,64,64,39,39,39]
    p.setJointMotorControlArray(robot,joint_vec,p.POSITION_CONTROL,
                                targetPositions=targetpos_vec,forces=force_vec,
                                positionGains = posGain)

def torque_control(p,robot,trq_cmd):
    joint_vec = [0,1,2,3,4,5,6]

    p.setJointMotorControlArray(robot,joint_vec,p.TORQUE_CONTROL,
                                forces=trq_cmd.tolist())

def velocity_control(p,robot,targetvel):
    for i in range(8):
        p.setJointMotorControl2(robot,i,p.VELOCITY_CONTROL,targetVelocity=targetvel,force=50)

# returns the cartesian pos of the ee in world frame
def FK(model,data,q,ee_id):
    
    pin.forwardKinematics(model,data,np.array(q))
    pin.updateFramePlacements(model,data)

    return data.oMf[ee_id].translation

##############################################################
# Create an instance of the Physics Server and connect to it
##############################################################

# Connect to the GUI client
client = p.connect(p.GUI)
if client < 0:
    raise Exception("Failed to connect to PyBullet GUI")

# Load the URDF of the plane that forms the ground
p.setAdditionalSearchPath(pybullet_data.getDataPath()) # Set the search path to find the plane.urdf file
plane = p.loadURDF("plane.urdf")

# robot import URDF
str1 = os.getcwd() + "/flexiv_rizon4_kinematics.urdf"
robot = p.loadURDF(str1,flags=p.URDF_USE_SELF_COLLISION,useFixedBase=1)

#------------------------------------------------#
#------- Pinocchio Model ------------------------#
#------------------------------------------------#

pin_model = pin.buildModelFromUrdf(str1)
pin_data = pin_model.createData()
ee_id = pin_model.getFrameId("flange")

##################################################
# Set the necessary parameters for the simulation
##################################################

# Set the Gravity vector
p.setGravity(0,0,-9.81, physicsClientId = client)

# Setting all the joints for velocity control    
for i in range(8):
    p.setJointMotorControl2(robot, i, p.VELOCITY_CONTROL, force=0)

torque_limits = np.array([400, 400, 100, 100, 45, 45, 25])

print(p.getBasePositionAndOrientation(robot))

timeStep =1/240 # 
p.setTimeStep(timeStep)

# Reset robot

reset_home(p,robot)

q_zero = np.zeros(p.getNumJoints(robot)-1)
q_initial  = np.deg2rad([0, -40, 0, 90, 0, 40, 0])
q_final  = np.deg2rad([0, -40, 0, 90, 0, 40, 0])

#----------------------------------------------

##---- Pose for IK----------------------------
ee_tgt_pos = [0.2,0.2,1.2] #x,y,z
ee_index_pyb = p.getNumJoints(robot) - 1  # Assuming the end-effector is the last link

q_tgt = p.calculateInverseKinematics(robot,ee_index_pyb,ee_tgt_pos)
qd_tgt = np.zeros(p.getNumJoints(robot)-1)

#-------- Traj gen using RRT ------------
# joint_indices = list(range(len(q_initial)))
# rrt = RRT.RRT(q_initial, q_final, robot, joint_indices)
# path = rrt.build_tree()

# if path is not None:
#     print("Found a path:")
#     for q in path:
#         print(q)
#         p.setJointMotorControlArray(robot, joint_indices, p.POSITION_CONTROL, q)
#         p.stepSimulation()
#         time.sleep(0.1)
# else:
#     print("No path found.")

#-----------------------------------------
# POSITION-PD CONTROL
#-----------------------------------------



# # plotting states
# # Initialize lists to store joint states and time
# nSteps = 500
# nq = p.getNumJoints(robot)-1
# q = np.zeros([nSteps,nq])
# qd = np.zeros([nSteps,nq])
# ee_pose = np.zeros([nSteps,3])
# ee_error = np.zeros([nSteps,3])

# t = np.zeros(nSteps)
# tbegin = time.time()

# posGains = [0.1 for _ in range(p.getNumJoints(robot)-1)] 
# print("posGain = ",posGains)

# for i in range (nSteps): 
#     position_control(p,robot,q_tgt,posGains)
#     p.stepSimulation()
#     q[i,:],qd[i,:] = get_all_joint_states(p,robot)

#     # Forward Kin to get the location of ee
#     ee_pose[i,:] = FK(pin_model,pin_data,q[i,:],ee_id)
#     ee_error[i,:] = abs(ee_pose[i,:]-np.array(ee_tgt_pos))


#     time.sleep(timeStep)
#     t[i] = time.time() - tbegin

# # Plotting states ----------------------------------#

# # plot_3d_positions(t,ee_pose)
# plot_3d_error(t,ee_error)

# plt.show()


#-----------------------------------------
# PID Torque control
#-----------------------------------------

# plotting states
# Initialize lists to store joint states and time
"""
nSteps = 1000
nq = p.getNumJoints(robot)-1
q = np.zeros([nSteps,nq])
qd = np.zeros([nSteps,nq])
tau = np.zeros([nSteps,nq])

ee_pose = np.zeros([nSteps,3])
ee_error = np.zeros([nSteps,3])

t = np.zeros(nSteps)
tbegin = time.time()

Kp = 1000.0 
Kd = 1.0
Ki = 10.0

q_err_int = np.zeros(p.getNumJoints(robot)-1)
q_err_prev = np.zeros(p.getNumJoints(robot)-1)
dt = 1.0

for i in range (nSteps): 

    
    q[i,:],qd[i,:] = get_all_joint_states(p,robot)

    q_err = q_tgt- q[i,:] # q_des - q_now
    q_err_int += q_err*timeStep    # accumulating error
    dq_err = (q_err - q_err_prev)/timeStep

    q_err_prev = q_err
    q_err_int = np.clip(q_err_int, -10.0, 10.0)

    trq_cmd = Kp*q_err  + Kd*dq_err + Ki*q_err_int
    trq_cmd = np.clip(trq_cmd, -torque_limits, torque_limits)
    tau[i,:] = trq_cmd

    torque_control(p,robot,trq_cmd)
    p.stepSimulation()

    # Forward Kin to get the location of ee
    ee_pose[i,:] = FK(pin_model,pin_data,q[i,:],ee_id)
    ee_error[i,:] = abs(ee_pose[i,:]-np.array(ee_tgt_pos))

    t_prev = time.time()

    time.sleep(timeStep)
    t[i] = time.time() - tbegin

# Plotting states ----------------------------------#

plot_3d_positions(t,ee_pose)
plot_3d_error(t,ee_error)
plot_trq(t,tau)

plt.show()

"""