import numpy as np
import pybullet as p
import time
import pinocchio as pin
import math
import gepetto.corbaserver
from time import sleep
import scipy

def SE3ToXYZQUATtuple(M):
    """Convert a pin.SE3 object into (x, y, z, qx, qy, qz, qw) for Gepetto"""
    from scipy.spatial.transform import Rotation
    pos = M.translation
    quat = Rotation.from_matrix(M.rotation).as_quat()  # Returns (qx, qy, qz, qw)
    return tuple(pos) + tuple(quat)

def print_model_info(model):
    print("model.name: ", model.name)
    print("model.njoints: ", model.njoints)
    print("model.nq: ", model.nq)
    print("model.nv = ", model.nv)
    print("Detail of each joint:")
    for name in model.names:
        print(name)
        print(model.getJointId(name))

    print("---------------------------")
    print("Frame Names")
    for frame in model.frames:
        print("frame.name: ", frame.name)
        print("frame.id: ", model.getFrameId(frame.name))

# Initialize the viewer
viewer = gepetto.corbaserver.Client()
viewer.gui.createWindow("window")
viewer.gui.createScene("scene")
viewer.gui.addSceneToWindow("scene", viewer.gui.getWindowID("window"))


# Load the robot model
urdf_path = "flexiv_rizon4_kinematics.urdf"
model = pin.buildModelFromUrdf(urdf_path)
data = model.createData()

# Add the robot to the viewer
robot_name = "robot"
viewer.gui.addURDF(robot_name, urdf_path)
viewer.gui.addToGroup(robot_name, "scene")


# print_model_info(model)
ee_frame_id = model.getFrameId("link7") # End-effector frame id

home_q = [0, -40, 0, 90, 0, 40, 0]  # Degrees
home_q_rad = np.array([math.radians(angle) for angle in home_q])
home_v_rad = np.random.rand(model.nv) # Random joint velocities
home_a_rad = np.random.rand(model.nv) # Random joint accelerations

print("home_q_rad: ", home_q_rad)

pin.forwardKinematics(model, data, home_q_rad, home_v_rad, home_a_rad) # FK with home position, rand v, rand a
pin.updateFramePlacements(model, data) # Update frame placements

viewer.gui.applyConfiguration(robot_name, SE3ToXYZQUATtuple(data.oMf[ee_frame_id]))
viewer.gui.refresh()


# Visualize the positions and velocities
for i in range(model.njoints):
    joint_name = model.names[i]
    joint_pos = data.oMi[i].translation
    joint_rot = data.oMi[i].rotation
    print(f"Joint {joint_name} position: {joint_pos}")
    print(f"Joint {joint_name} rotation: {joint_rot}")

    # Update the viewer with the joint positions
    viewer.gui.applyConfiguration(joint_name, pin.SE3ToXYZQUATtuple(data.oMi[i]))
    viewer.gui.refresh()
    sleep(0.1)

# Visualize the end-effector position and velocity
ee_frame_id = model.getFrameId("link7")
ee_pos = data.oMf[ee_frame_id].translation
ee_rot = data.oMf[ee_frame_id].rotation
ee_vel = pin.getFrameVelocity(model, data, ee_frame_id, pin.WORLD)
print("End-effector position: ", ee_pos)
print("End-effector rotation: ", ee_rot)
print("End-effector velocity: ", ee_vel)

# getting joint IDs
joint_id_vec = []

for joint in model.names:
    joint_id_vec.append(model.getJointId(joint))

print("joint id vec = ", joint_id_vec)

for i in range(0,model.njoints):
    print(" i = ", i)
    print("pos of joint = ", data.oMi[i].translation)
    print("rot of joint = ", data.oMi[i].rotation)


#------------------------
# EE position/velocity + rotation
#------------------------

ee_pos = data.oMf[ee_frame_id].translation
ee_rot = data.oMf[ee_frame_id].rotation
print("ee_pos: \n", ee_pos)
print("ee_rot: \n", ee_rot)

ee_vel = pin.getFrameVelocity(model, data, ee_frame_id, pin.WORLD)
print("ee_vel: ", ee_vel)

## verifying that vee = J(q) * q_dot
J = pin.computeFrameJacobian(model, data, home_q_rad, ee_frame_id, pin.WORLD) # only works in the world frame

J_v = J @ home_v_rad

# difference between vee and J(q) * q_dot 
print(ee_vel.linear - J_v[:3])
print(ee_vel.angular - J_v[3:])

