 
import pinocchio as pin
import numpy as np
np.set_printoptions(suppress=True, precision=6)
from scipy.spatial.transform import Rotation as Rot

def computeFK(model, data, q, ee_frame_id):
    """
    Compute Forward Kinematics for the given model and configuration.
    """
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    return data.oMf[ee_frame_id].translation

def computeJac(model, data, q, ee_frame_id):
    """
    Compute the Jacobian for the end-effector frame.
    """
    J = pin.computeFrameJacobian(model, data, q, ee_frame_id, pin.ReferenceFrame.WORLD)
    J_pos = J[0:3, :]  # Only position part of the Jacobian
    J_rot = J[3:6, :]  # Rotation part of the Jacobian
    return J_pos

def computeJinv(model, data, q, ee_frame_id):
    """
    Compute the inverse of the Jacobian for the end-effector frame.
    """
    J = computeJac(model, data, q, ee_frame_id)
    J_inv = np.linalg.pinv(J)  # Pseudo-inverse of the Jacobian
    return J_inv

def IK(model, data, desired_ee_pos, ee_frame_id, q_guess=None, max_iterations=100000, tolerance=1e-6):
    """
    Perform Inverse Kinematics to find joint angles that achieve the desired end-effector position.
    """
    print("Running Inverse Kinematics...")
    if q_guess is None:
        q_guess = np.zeros(model.nq)  # Start from zero configuration

    q = np.copy(q_guess)
    for i in range(max_iterations):
        current_ee_pos = computeFK(model, data, q, ee_frame_id)
        error = desired_ee_pos - current_ee_pos

        if np.linalg.norm(error) < tolerance:
            print(f"Converged in {i} iterations.")
            return q

        J_inv = computeJinv(model, data, q, ee_frame_id)
        delta_q = J_inv @ error
        q += delta_q

    print("Did not converge within the maximum number of iterations.")
    print("Final position:", computeFK(model, data, q, ee_frame_id))
    return q

# Load the robot model
urdf_path = "flexiv_rizon4_kinematics.urdf"
model = pin.buildModelFromUrdf(urdf_path)
print("-----------------------------------")
print("-----------------------------------")

print("model name = ", model.name)
print("Joint Names, note that the zeroth joint is universe joint")
for jt in model.names:
    print(jt)
print("model.njoints = ", model.njoints)
print("model.nv = ", model.nv)
print("model.nq = ", model.nq)

print("-----------------------------------")
print("-----------Frames------------------")
print("-----------------------------------")
print("model.nframes = ", model.nframes)
for i in range(model.nframes):
    print(model.frames[i].name)

print("Create data")
data = model.createData()

# Set the configuration of the robot
q = pin.randomConfiguration(model)
print("\n q = ", q)

print("size of data.oMi = ", len(data.oMi))
print("size of data.oMf = ", len(data.oMf))
ee_frame_id = model.getFrameId("flange")  # End-effector frame id
print("ee_frame_id = ", ee_frame_id)

# Forward kinematics
print("\n Forward Kinematics")
fk_position = computeFK(model, data, q, ee_frame_id)
print("fk_position = ", fk_position)

# Compute the Jacobian
print("\n Compute Jacobian")
jacobian = computeJac(model, data, q, ee_frame_id)
print("Jacobian (position part):", jacobian)

# Compute the inverse of the Jacobian
print("\n Compute Inverse Jacobian")
jacobian_inv = computeJinv(model, data, q, ee_frame_id)
print("Inverse Jacobian:", jacobian_inv)

# set a desired ee pos
desired_ee_pos = np.array([0.1, 0.1, 0.1])  # Example desired position
print("\n Desired End-Effector Position:", desired_ee_pos)
# Perform Inverse Kinematics

q_final = IK(model, data, desired_ee_pos, ee_frame_id)
print("\n Final Joint Configuration:", q_final)
