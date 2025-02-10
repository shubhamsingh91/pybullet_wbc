import numpy as np
import pybullet as p
import time
import cvxpy as cp
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.DEBUG)

# Global lists for storing history
task_cost_history = []
time_steps = []
pos_x_history, pos_y_history, pos_z_history = [], [], []
ref_x_history, ref_y_history, ref_z_history = [], [], []


def update_plots(fig, axs):
    """
    Updates the matplotlib plots for task cost and position tracking.
    """
    axs[0].clear()
    axs[0].plot(time_steps, task_cost_history, 'b-')
    axs[0].set_title("Task Cost (Log Scale)")
    axs[0].set_ylabel("Task Cost (log)")
    axs[0].set_yscale("log")

    # Update X Position Plot
    axs[1].clear()
    axs[1].plot(time_steps, ref_x_history, 'r--', label="Reference X")
    axs[1].plot(time_steps, pos_x_history, 'b-', label="Current X")
    axs[1].set_title("X Position Tracking")
    axs[1].legend()

    # Update Y Position Plot
    axs[2].clear()
    axs[2].plot(time_steps, ref_y_history, 'r--', label="Reference Y")
    axs[2].plot(time_steps, pos_y_history, 'b-', label="Current Y")
    axs[2].set_title("Y Position Tracking")
    axs[2].legend()

    # Update Z Position Plot
    axs[3].clear()
    axs[3].plot(time_steps, ref_z_history, 'r--', label="Reference Z")
    axs[3].plot(time_steps, pos_z_history, 'b-', label="Current Z")
    axs[3].set_title("Z Position Tracking")
    axs[3].legend()

    # Adjust layout and refresh
    plt.tight_layout()
    plt.pause(0.1)  # Pause for 100ms to update the plot


# returns xddot_des in task space   
# xddot = kp*(x_des - x) + kd*(v_des - v)             
def compute_task_desired_accel(pos_des, pos_cur, vel_cur, vel_des,kp, kd=100.0):
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
    vel_error = vel_des-vel_cur 

    # PD acceleration
    accel_des = np.array(kp)*np.array(pos_error) + np.array(kd)*np.array(vel_error)
    return accel_des

def get_end_effector_pose(robot_id, end_eff_link):
    # returns (position, orientation) of the link in world frame
    state = p.getLinkState(robot_id, end_eff_link)
    pos  = state[0]  # linkWorldPosition
    orn  = state[1]  # linkWorldOrientation
    return np.array(pos), np.array(orn)

def get_end_effector_velocity(robot_id, end_eff_link):
    # returns (linear velocity, angular velocity) of the link in world frame
    state = p.getLinkState(robot_id, end_eff_link, computeLinkVelocity=1)
    lin_vel = state[6]  # linkWorldLinearVelocity
    ang_vel = state[7]  # linkWorldAngularVelocity
    return np.array(lin_vel), np.array(ang_vel)

def get_jacobian(robot_id, dof, end_eff_link, q, q_dot):
    # linkCoM in local coordinates (0,0,0) if you want the Jacobian at the link frame origin
    # or some offset if your end-effector is offset from the link frame.
    zero_vec = [0.0]*dof
    jac_t, jac_r = p.calculateJacobian(robot_id, end_eff_link, [0,0,0], 
                                       q.tolist(), q_dot.tolist(), zero_vec)
    # jac_t, jac_r are each 3 x DOF
    # Combine them if you also care about orientation tasks. For now, let's do position only.
    jac_t = np.array(jac_t)  # shape (3, DOF)
    jac_r = np.array(jac_r)  # shape (3, DOF)
    # If needed, orientation part is in jac_r (3, DOF)
    return jac_t, jac_r

def get_joint_state(p,robot,joint):
    temp1 =p.getJointState(robot,joint)   
    return temp1[0],temp1[1], temp1[2:6]     


def get_all_joint_states(p,robot,dof):
    q=[0]*dof
    qd=[0]*dof
    jointReactionForces=[0]*6
    tau = [0]*dof
    for i in range(7):
        q[i], qd[i],jointReactionForces = get_joint_state(p, robot, i)
    return q,qd    

def wbc_qp_step(robot_id, dof, end_eff_link, pos_des, vel_des, q_ddot_min=-100, q_ddot_max=100, alpha=1e-3):
    """
    Solve a minimal QP for a single task: position control of the end-effector.
    Return: tau (joint torques) to apply
    """
    # 1) Get current joint states
    q_list = []
    qd_list = []
    for j in range(dof):
        joint_info = p.getJointState(robot_id, j)
        q_list.append(joint_info[0])
        qd_list.append(joint_info[1])
    q = np.array(q_list)
    q_dot = np.array(qd_list)
    
    # 2) Get current end-effector pos, velocity
    pos_cur, _ = get_end_effector_pose(robot_id, end_eff_link)
    vel_cur, _= get_end_effector_velocity(robot_id, end_eff_link)
    
    # 3) Desired task-space accel
    kp = [500.0, 500.0, 500.0]  # PD gains
    x_ddot_des = compute_task_desired_accel(pos_des, pos_cur, vel_cur, vel_des, kp, kd=10.0)
    
    # 4) Build J, dJ, etc.
    jac_t, jac_r = get_jacobian(robot_id, dof, end_eff_link, q, q_dot)
    # Approx dJ * q_dot by numerical difference or using PyBullet (not directly provided)
    # For a simpler hack, assume it's small => dJdot = 0
    dJ_qdot = np.zeros(3) 
    
    # 5) QP variables
    qddot_var = cp.Variable(dof)
    
    # Task-space cost:  ||J q_ddot + dJdot - x_ddot_des||^2

    task_resid = jac_t @ qddot_var + dJ_qdot - x_ddot_des

    task_cost = cp.sum_squares(task_resid)
    
    # Regularization on q_ddot
    reg_cost = alpha * cp.sum_squares(qddot_var)
    
    obj = cp.Minimize(task_cost+reg_cost)
    
    # Constraints
    constraints = []
    constraints.append(qddot_var >= q_ddot_min)
    constraints.append(qddot_var <= q_ddot_max)
    
    prob = cp.Problem(obj, constraints)

    # result = prob.solve(verbose=True)
    result = prob.solve()

    if qddot_var.value is None:
        # Solver failed
        print("QP solver failed!")
        return np.zeros(dof)
    else:
        print("QP solver success!")
        print("qddot_var:", qddot_var.value)
        print("cost:", result)
    
    q_ddot_sol = qddot_var.value
    

    # 6) Convert q_ddot_sol to torque using inverse dynamics
    # We'll use PyBullet's built-in function for convenience:
    # p.calculateInverseDynamics(robot_id, q, q_dot, q_ddot_sol)
    #   returns a list of joint torques that would achieve those accelerations
    tau = p.calculateInverseDynamics(robot_id, q.tolist(), q_dot.tolist(), q_ddot_sol.tolist())
    tau = np.array(tau)
    
    return tau, result, pos_cur

def main_control_loop(robot_id, dof, end_eff_link, pos_des, vel_des):
    # For example, we want to keep the end-effector at [0.5, 0.0, 0.5] in world frame
    print("-----Main Control Loop-----------------")
    
    # Initialize live plotting
    plt.ion()
    fig, axs = plt.subplots(4, 1, figsize=(8, 12))  # 4 Subplots

    step = 0  # Counter for time steps

    while True:
        # 1) Solve WBC QP step
        print("--------------------------------------")
        tau, task_cost, pos_cur = wbc_qp_step(robot_id, dof, end_eff_link, pos_des,vel_des)
        print("tau:", tau)
        
        # 2) Apply joint torques
        # PyBullet's setJointMotorControl2 with CONTROL_TORQUE mode per joint
        for j in range(dof):
            p.setJointMotorControl2(bodyUniqueId=robot_id, 
                                    jointIndex=j, 
                                    controlMode=p.TORQUE_CONTROL, 
                                    force=tau[j])
        # 3) Step simulation
        p.stepSimulation()

        pos_cur, orn = get_end_effector_pose(robot_id, end_eff_link)
        vel_cur, ang_vel = get_end_effector_velocity(robot_id, end_eff_link) 
        print("Current position:", pos_cur)
        print("desired position:", pos_des)
        print("Current velocity:", vel_cur)
        print("desired velocity:", vel_des)

        # Store values for plotting
        if task_cost is not None:
            task_cost_history.append(task_cost)
            time_steps.append(step)

            pos_x_history.append(pos_cur[0])
            pos_y_history.append(pos_cur[1])
            pos_z_history.append(pos_cur[2])

            ref_x_history.append(pos_des[0])
            ref_y_history.append(pos_des[1])
            ref_z_history.append(pos_des[2])

            step += 1

        # **Only plot every 100 steps**
            if step % 100 == 0:
                update_plots(fig, axs)