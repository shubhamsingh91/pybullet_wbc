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
    print("Jacobian:", jac_t)
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

def wbc_qp_step(robot_id, dof, end_eff_link, pos_des, q_ddot_min=-10, q_ddot_max=10, alpha=1e-2):
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
    
    print("Step -2")
    # 2) Get current end-effector pos, velocity
    pos_cur, _ = get_end_effector_pose(robot_id, end_eff_link)
    vel_cur, _= get_end_effector_velocity(robot_id, end_eff_link)
    
    print("Step -3")
    # 3) Desired task-space accel
    x_ddot_des = compute_task_desired_accel(pos_des, pos_cur, vel_cur, kp=50.0, kd=10.0)
    
    print("Step -4")
    # 4) Build J, dJ, etc.
    jac_t, jac_r = get_jacobian(robot_id, dof, end_eff_link, q, q_dot)
    print("Jacobian:", jac_t)
    # Approx dJ * q_dot by numerical difference or using PyBullet (not directly provided)
    # For a simpler hack, assume it's small => dJdot = 0
    dJ_qdot = np.zeros(3) 
    
    print("Step -5")
    # 5) QP variables
    qddot_var = cp.Variable(dof)
    print("x_ddot_des:", x_ddot_des)
    print("qddot_var:", qddot_var)
    print('dJ_qdot:', dJ_qdot)
    
    # Task-space cost:  ||J q_ddot + dJdot - x_ddot_des||^2
    task_resid = jac_t @ qddot_var + dJ_qdot - x_ddot_des
    task_cost = cp.sum_squares(task_resid)
    print("task_cost:", task_cost)
    print("task_resid:", task_resid)
    
    # Regularization on q_ddot
    reg_cost = alpha * cp.sum_squares(qddot_var)
    
    obj = cp.Minimize(task_cost + reg_cost)
    
    # Constraints
    constraints = []
    constraints.append(qddot_var >= q_ddot_min)
    constraints.append(qddot_var <= q_ddot_max)
    
    prob = cp.Problem(obj, constraints)
    print("--------------------------------------")
    print("solving")

    result = prob.solve()  # solve the QP
    
    if qddot_var.value is None:
        # Solver failed
        print("QP solver failed!")
        return np.zeros(dof)
    
    q_ddot_sol = qddot_var.value
    
    # 6) Convert q_ddot_sol to torque using inverse dynamics
    # We'll use PyBullet's built-in function for convenience:
    # p.calculateInverseDynamics(robot_id, q, q_dot, q_ddot_sol)
    #   returns a list of joint torques that would achieve those accelerations
    tau = p.calculateInverseDynamics(robot_id, q, q_dot, q_ddot_sol)
    tau = np.array(tau)
    
    return tau

def main_control_loop(robot_id, dof, end_eff_link, pos_des):
    # For example, we want to keep the end-effector at [0.5, 0.0, 0.5] in world frame
    print("-----Main Control Loop-----------------")
    
    while True:
        # 1) Solve WBC QP step
        print("Solving WBC QP step")
        tau = wbc_qp_step(robot_id, dof, end_eff_link, pos_des)
        
        # 2) Apply joint torques
        # PyBullet's setJointMotorControl2 with CONTROL_TORQUE mode per joint
        for j in range(dof):
            p.setJointMotorControl2(bodyUniqueId=robot_id, 
                                    jointIndex=j, 
                                    controlMode=p.TORQUE_CONTROL, 
                                    force=tau[j])
        print("Joint torques calculation:")
        # 3) Step simulation
        p.stepSimulation()

        pos_cur, orn = get_end_effector_pose(robot_id, end_eff_link)
        vel_cur, ang_vel = get_end_effector_velocity(robot_id, end_eff_link) 
        print("Current position:", pos_cur)
        print("Current velocity:", vel_cur)
