import pybullet as p
import pybullet_data
import numpy as np
import time
import wbc as wbc

# If you want a small QP solver in Python, you can use cvxpy or quadprog.
# For example: 
#   pip install cvxpy
import cvxpy as cp

# For demonstration, let's assume:
END_EFFECTOR_LINK = 7    # The link index for the end-effector in PyBullet

client1 = p.DIRECT
client = p.GUI # or p.GUI
p.connect(p.GUI)
p.connect(client1)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load the plane and the manipulator (e.g., KUKA robot)
p.loadURDF("plane.urdf")
robot_id = p.loadURDF("flexiv_rizon4_kinematics.urdf", useFixedBase=True)
p.setGravity(0, 0, -9.81)

# Get the number of joints
num_joints = p.getNumJoints(robot_id)
print("\nnum_joints = ", num_joints)

#Compute DoF by counting movable joints
dof = sum(1 for i in range(num_joints) if p.getJointInfo(robot_id, i)[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC])

print(f"Degrees of Freedom (DoF): {dof}")


# Print joint info
for joint_index in range(num_joints):
    info = p.getJointInfo(robot_id, joint_index)
    print(info)

print("Robot ID:", robot_id)
# print link ids




def set_pos_control(des_pos):
    for joint_index in range(len(des_pos)):
            p.setJointMotorControl2(bodyIndex=robot_id,
                                    jointIndex=joint_index,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=des_pos[joint_index])


pos, orn = wbc.get_end_effector_pose(robot_id, END_EFFECTOR_LINK)
print("End-effector position:", pos)
print("End-effector orientation:", orn)

q = np.random.uniform(-1.0, 1.0, size=dof).tolist()
q_dot = np.random.uniform(-1.0, 1.0, size=dof).tolist()
print("Joint positions:", q)
print("Joint velocities:", q_dot)

jac_t, jac_r = wbc.get_jacobian(robot_id, END_EFFECTOR_LINK, 
                                       q, q_dot, dof)
set_pos_control(q)
print("Jacobian t:", jac_t)



for i in range(10000):
    p.stepSimulation()
    time.sleep(1./240.)
    # input("Press Enter to exit...")  # Wait for user input
    # p.disconnect()  # Close PyBullet


