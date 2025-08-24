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

def set_pos_control(des_pos):
    for joint_index in range(len(des_pos)):
            p.setJointMotorControl2(bodyIndex=robot_id,
                                    jointIndex=joint_index,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=des_pos[joint_index])

# make pinocchio model and run IK
urdf_path = "flexiv_rizon4_kinematics.urdf"
model = pin.buildModelFromUrdf(urdf_path)
data = model.createData()
ee_frame_id = model.getFrameId("flange")  # End-effector frame id

q_home = np.array([0, -40, 0, 90, 0, 40, 0], dtype=np.float64)  # Home configuration in radians

# start simulation at home, take user input for x_des , run IK and go to pose
p.setRealTimeSimulation(1)
while True:
    p.stepSimulation()
    time.sleep(0.01)  # Sleep to control simulation speed
    
    # Get user input for desired end-effector position
    x_des_input = input("Enter desired end-effector position as x,y,z (or 'exit' to quit): ")
    
    if x_des_input.lower() == 'exit':
        break
    
    try:
        x_des = np.array([float(coord) for coord in x_des_input.split(',')], dtype=np.float64)
        q_final = IK(model, data, x_des, ee_frame_id, q_home)
        set_pos_control(q_final)
        print(f"Moving to desired position: {x_des}")
    except Exception as e:
        print(f"Error: {e}. Please enter valid coordinates.")

# Disconnect
p.disconnect()
