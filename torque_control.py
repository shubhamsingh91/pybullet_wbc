import pybullet as p, pybullet_data as pd
import numpy as np, time
import pinocchio as pin
from IK_test import IK

cid = p.connect(p.GUI)
p.setAdditionalSearchPath(pd.getDataPath())
p.resetSimulation(physicsClientId=cid)
p.setGravity(0,0,-9.81, physicsClientId=cid)
p.setTimeStep(1/240.0, physicsClientId=cid)
p.setRealTimeSimulation(0, physicsClientId=cid)

p.setAdditionalSearchPath(pd.getDataPath())

# Load the plane and the manipulator (e.g., KUKA robot)
p.loadURDF("plane.urdf")
robot_id = p.loadURDF("flexiv_rizon4_kinematics.urdf", useFixedBase=True)

# 1) Actuated joints only
actuated = []
for j in range(p.getNumJoints(robot_id, physicsClientId=cid)):
    jtype = p.getJointInfo(robot_id, j, physicsClientId=cid)[2]
    if jtype in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
        actuated.append(j)

# 3) Read URDF effort limits for torque clamp
tau_max = []
for j in actuated:
    info = p.getJointInfo(robot_id, j, physicsClientId=cid)
    effort = info[10] if info[10] > 0 else 1e6  # fallback if 0
    tau_max.append(effort)
tau_max = np.array(tau_max, float)
tau_min = -tau_max

print("tau_max:", tau_max)
print("tau_min:", tau_min)

print("Actuated joints:", actuated)

# Get the number of joints
num_joints = p.getNumJoints(robot_id)
print("\nnum_joints = ", num_joints)

# Print joint info
for joint_index in range(num_joints):
    info = p.getJointInfo(robot_id, joint_index)
    jtype = info[2]
    print(info)
    print("Joint index:", joint_index, "Type:", jtype)
    
def get_joint_states():
    '''
    Get the current joint positions and velocities.'''
    st = p.getJointStates(robot_id, actuated, physicsClientId=cid)
    q  = np.array([s[0] for s in st], float)
    qd = np.array([s[1] for s in st], float)
    reaction_wrench = [s[2] for s in st]  # [Fx,Fy,Fz,Mx,My,Mz] in world frame
    joint_moments = [w[3:] for w in reaction_wrench]  # use Mx/My/Mz as “felt” torques
    # print("joint moments (Mx,My,Mz):", [np.round(m,2) for m in joint_moments])
    tau_meas = np.array([s[3] for s in st], float)  # Measured joint torques
    return q, qd, tau_meas

def set_pos_control(des_pos):

    for j in actuated:
            p.setJointMotorControl2(bodyIndex=robot_id,
                                    jointIndex=j,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=des_pos[j])

def disable_motor_control():
    for j in actuated:
        p.setJointMotorControl2(robot_id, j, p.VELOCITY_CONTROL, force=0, physicsClientId=cid)

    # enable joint torque sensors
    for j in actuated:
        p.enableJointForceTorqueSensor(robot_id, j, enableSensor=1, physicsClientId=cid)
    # set zero damping and friction for all joints
    for j in range(p.getNumJoints(robot_id, physicsClientId=cid)):
        p.changeDynamics(robot_id, j,
                        linearDamping=0.0,
                        angularDamping=0.0,
                        lateralFriction=0.2,
                        rollingFriction=0.0,
                        spinningFriction=0.0,
                        physicsClientId=cid)


def apply_torque_limited(tau):
    tau = np.clip(tau, tau_min, tau_max)
    p.setJointMotorControlArray(robot_id, actuated, controlMode=p.TORQUE_CONTROL,
                                forces=tau.tolist(), physicsClientId=cid)

def grav_comp_bullet(q):
    zeros = [0.0]*len(q)
    return np.array(p.calculateInverseDynamics(robot_id, list(q), zeros, zeros, physicsClientId=cid))


def grav_comp(pin_model,pin_data,q,qd):
    '''
    Compute gravity compensation torques for the given joint configuration q.
    '''
    # pin_model.gravity.linear = np.array([0,0,-9.81])
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


disable_motor_control()  # Disable default motor control to allow torque control

# Kp and Kd for PD control
n = len(actuated)
Kp = 40.0 * np.ones(n)
Kd = 2.0 * np.sqrt(Kp)   # near-critical damping
qd_des = np.zeros(n)
alpha = 0.2
qd_filt = np.zeros(n)
print("q_des = ", q_des)

while True:

    q, qdot, tau_meas  = get_joint_states()
    qd_filt = (1-alpha)*qd_filt + alpha*qdot
    
    print("\n-------------------")
    # print current joint states upto 2 decimal places
    print("Current Joint Positions: ", np.round(q,2))
    print("Current Joint Velocities filt: ", np.round(qd_filt,2))

    tau_g_bullet = grav_comp_bullet(q)
    print("tau_g_bullet = ", np.round(tau_g_bullet,2))

    # # PD control torques
    tau_pd = Kp.dot(q_des - q) + Kd.dot(qd_des - qd_filt) + tau_g_bullet

    print("q = ", np.round(q,2))
    print("q_des = ", np.round(q_des,2))

    # Apply torque control

    apply_torque_limited(tau_g_bullet)
    # set_pos_control(q_des)  # Optional: for smoother behavior

    p.stepSimulation(physicsClientId=cid)
    time.sleep(1/240.0)

# Disconnect
p.disconnect()
