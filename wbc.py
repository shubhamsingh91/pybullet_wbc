import numpy as np
import pybullet as p
import time
import cvxpy as cp

# returns xddot_des in task space   
# xddot = kp*(x_des - x) + kd*(v_des - v)             
def compute_task_desired_accel(pos_des, pos_cur, vel_cur, kp=50.0, kd=10.0):
    """
    pos_des: desired 3D position
    pos_cur: current 3D position
    vel_cur: current 3D velocity of the end-effector (approx)
    kp, kd : PD gains
    Return: 3D acceleration vector in task space
    """
    pos_error = pos_des - pos_cur
    # We treat vel_cur as the end-effector's linear velocity in world frame
    # If you have a desired velocity, you can do e_vel = vel_des - vel_cur
    # For simplicity, assume desired vel = 0
    vel_error = -vel_cur 

    # PD acceleration
    accel_des = kp*pos_error + kd*vel_error
    return accel_des

def get_end_effector_pose(robot_id, end_eff_link):
    # returns (position, orientation) of the link in world frame
    state = p.getLinkState(robot_id, end_eff_link)
    pos  = state[0]  # linkWorldPosition
    orn  = state[1]  # linkWorldOrientation
    return np.array(pos), np.array(orn)

def get_jacobian(robot_id, end_eff_link, q, q_dot,dof):
    # linkCoM in local coordinates (0,0,0) if you want the Jacobian at the link frame origin
    # or some offset if your end-effector is offset from the link frame.
    zero_vec = [0.0]*dof
    jac_t, jac_r = p.calculateJacobian(robot_id, end_eff_link, [0,0,0], 
                                       q, q_dot, zero_vec)
    # jac_t, jac_r are each 3 x DOF
    # Combine them if you also care about orientation tasks. For now, let's do position only.
    jac_t = np.array(jac_t)  # shape (3, DOF)
    jac_r = np.array(jac_r)  # shape (3, DOF)
    # If needed, orientation part is in jac_r (3, DOF)
    return jac_t, jac_r
