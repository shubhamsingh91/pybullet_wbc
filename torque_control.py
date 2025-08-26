import pybullet as p
import pybullet_data
import time
from IK_test import IK
import pinocchio as pin
import numpy as np

# Connect to PyBullet
client1 = p.DIRECT
client = p.GUI # or p.GUI
p.connect(p.GUI)
p.connect(client1)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load the plane and the manipulator (e.g., KUKA robot)
p.loadURDF("plane.urdf")
robot_id = p.loadURDF("flexiv_rizon4_kinematics.urdf", useFixedBase=True)

# Set gravity
p.setGravity(0,0,-9.81, physicsClientId = client)

# Get the number of joints
num_joints = p.getNumJoints(robot_id)
print("\nnum_joints = ", num_joints)

# Print joint info
for joint_index in range(num_joints):
    info = p.getJointInfo(robot_id, joint_index)
    print(info)
    
def get_joint_states():
    '''
    Get the current joint positions and velocities.'''
    joint_states = p.getJointStates(robot_id, range(num_joints-1))
    joint_positions = [state[0] for state in joint_states]
    joint_velocities = [state[1] for state in joint_states]
    return joint_positions, joint_velocities

def set_pos_control(des_pos):
    for joint_index in range(len(des_pos)):
            p.setJointMotorControl2(bodyIndex=robot_id,
                                    jointIndex=joint_index,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=des_pos[joint_index])

def disable_motor_control():
    '''
    Disable motor control for all joints to allow torque control.'''
    for joint_index in range(num_joints):
            p.setJointMotorControl2(bodyIndex=robot_id,
                                    jointIndex=joint_index,
                                    controlMode=p.VELOCITY_CONTROL,
                                    force=0)

def torque_control(torques):
    '''
    Apply torque control to all joints.'''
    for joint_index in range(len(torques)):
            p.setJointMotorControl2(bodyIndex=robot_id,
                                    jointIndex=joint_index,
                                    controlMode=p.TORQUE_CONTROL,
                                    force=torques[joint_index])
            
def grav_comp(pin_model,pin_data,q,qd):
    '''
    Compute gravity compensation torques for the given joint configuration q.
    '''
    g = np.array([0, 0, -9.81])  # Gravity vector
    pin.computeAllTerms(pin_model, pin_data, np.copy(q), np.copy(qd))
    return pin_data.g

# make pinocchio pin_model and run IK
urdf_path = "flexiv_rizon4_kinematics.urdf"
pin_model = pin.buildModelFromUrdf(urdf_path)
pin_data = pin_model.createData()
ee_frame_id = pin_model.getFrameId("flange")  # End-effector frame id

q_home = np.zeros(pin_model.nv, dtype=np.float64)  # Home configuration in radians
x_des = np.array([0.1, 0.2, 0.5], dtype=np.float64)  # Desired end-effector position
q_des = IK(pin_model, pin_data, x_des, ee_frame_id, q_home)

print("q_des = ", q_des)
qd_des = np.zeros(pin_model.nv)

# start simulation at home, take user input for x_des , run IK and go to pose
p.setRealTimeSimulation(1)
disable_motor_control()  # Disable default motor control to allow torque control

# Kp and Kd for PD control
Kp = np.diag([50]*pin_model.nv)
Kd = np.diag([2]*pin_model.nv)

while True:
    p.stepSimulation()
    time.sleep(0.01)  # Sleep to control simulation speed

    q,qdot  = get_joint_states()
    # print current joint states upto 2 decimal places
    print("Current Joint Positions: ", np.round(q,2))
    print("Current Joint Velocities: ", np.round(qdot,2))

    # calculating gravity compensation torques
    tau_g = grav_comp(pin_model, pin_data, q, qdot)

    # PD control torques
    tau_pd = Kp.dot(q_des - q) + Kd.dot(qd_des - qdot) + tau_g

    print("Gravity Compensation Torques: ", np.round(tau_g,2))
    print("q_des - q = ", np.round(q_des - q,2))
    print("qdot_des - qdot = ", np.round(qd_des - qdot,2))
    print("PD Control Torques: ", np.round(tau_pd,2))
    # Apply torque control
    torque_control(tau_pd)
    
# Disconnect
p.disconnect()
