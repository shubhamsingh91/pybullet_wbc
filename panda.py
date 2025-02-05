import pybullet as p
import pybullet_data
import time

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


# Run the simulation for a while
# position control            
for i in range(10000):
    des_pos =  [1.57+i*0.001,1.57,1.571,1.57,1.57,1.57+i*0.001,i*0.001]
    set_pos_control(des_pos)
    p.stepSimulation()
    time.sleep(1./240.)

# Disconnect
p.disconnect()
