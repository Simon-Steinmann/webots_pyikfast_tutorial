# Copyright 2020 Simon Steinmann
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import numpy as np
from controller import Supervisor, Node

# import custom scripts. Located in the same directory as this controller
from get_relative_position import RelativePositions
from pyikfast_module import inverseKinematics
from spawn_target import spawnTarget

from transforms3d.axangles import mat2axangle

# ------------------- CONFIGURATION -------------------------

# how many simulationsteps before calculating the next IK solution.
# For velocity control, a smaller value makes the movement smoother.
IKstepSize = 2

supervisor = Supervisor()

# This name is needed in the pyikfast_module.py to determine the joint2_offset (line 25)
protoName = supervisor.getSelf().getTypeName()  #'UR10e' #'Irb4600-40'

# import the pyikfast solver module for your robot as "pyikfast_module"
# import pyikfast_irb4600_40 as pyikfast_module
if protoName == "UR10e":
    import pyikfast_ur10e as pyikfast_module
else:
    import pyikfast_irb4600_40 as pyikfast_module
print("pyikfast_module", pyikfast_module)

# Offset from toolSlot base, for which the IK solution is calculated.
# This can be useful, when attaching grippers on the robot.
# Offset = 0.2 is a fitting value for the UR10e with robotiq-3f gripper.
# Adding a "Transform" node to the toolSlot of the robot can be useful
# for figuring out how to adjust this offset.
offset = [0, 0, 0]
# -------------------- INITIALIZATION -----------------------

# Initialize the Webots Supervisor.
print("getTypeName", supervisor.getSelf().getTypeName())
if not supervisor.getSupervisor():
    sys.exit(
        "WARNING: Your robot is not a supervisor! Set the supervisor field to True and restart the controller."
    )
timeStep = int(supervisor.getBasicTimeStep())

# --------------------------------------------------------------------
# Initialize the arm motors and sensors. This is a generic code block
# and works with any robotic arm.
n = supervisor.getNumberOfDevices()
motors = []
sensors = []
minPositions = []
maxPositions = []
for i in range(n):
    device = supervisor.getDeviceByIndex(i)
    print(device.getName(), "   - NodeType:", device.getNodeType())
    # if device is a rotational motor (uncomment line above to get a list of all robot devices)
    if device.getNodeType() == Node.__dict__["ROTATIONAL_MOTOR"]:
        motors.append(device)
        minPositions.append(device.getMinPosition())
        maxPositions.append(device.getMaxPosition())
        sensor = device.getPositionSensor()
        try:
            sensor.getName()
            sensors.append(sensor)
            sensor.enable(timeStep)
        except Exception as e:
            print("Rotational Motor: " + device.getName() + " has no Position Sensor")
print("minPositions", minPositions)
print("maxPositions", maxPositions)

# --------------------------------------------------------------------
# move the arm onto target using known solution
target_position_joint_state = np.array([0, -np.pi / 2, np.pi / 2, 0, 0, 0]).astype(
    np.double
)
def move_to_joint_state(motors, joint_state):
    """Move robot endpoint to a known position that aligns it perfectly with the target"""
    for i in range(len(joint_state)):
        motors[i].setPosition(joint_state[i])
# plase end-effector on target
move_to_joint_state(motors, target_position_joint_state)

# --------------------------------------------------------------------
# show that the endpoint pose is indeed the targtet pose
# Initialize our inverse kinematics module
ik = inverseKinematics(pyikfast_module, protoName, minPositions, maxPositions)
print("ik", ik)

fk_pos, fk_rot = ik.get_fk(target_position_joint_state)
print("fk_pos", fk_pos)
print("fk_rot", fk_rot)

# check that the target position and rotation matches te end-effector's
target = supervisor.getFromDef("TARGET")
# target position relative to world.
target_pos_world = np.array(target.getPosition())
# target rotation relative to world
target_rot_world = np.array(target.getOrientation()).reshape(3, 3)
assert np.allclose(fk_pos, target_pos_world)
assert np.allclose(fk_rot, target_rot_world)
print("Yay, target pose and end-effector pose match")

# --------------------------------------------------------------------
# Try is get an IK solution for the current end-point pose.
# there should be at least one solution, target_position_joint_state
ikResults = ik.get_ik(fk_pos, fk_rot)

assert len(ikResults) > 0

#############################################################################################
# We don't get any further
print("Yay, We found in IK solution to our current end effector pose")


# Initialize the RelativePositions module. 'TARGET' is the DEF of the spawned sphere.
# You can change this DEF to any other object. You can also initialilze several like this:
# RelPos_1 = RelativePositions(supervisor, 'TARGET1')
# RelPos_2 = RelativePositions(supervisor, 'TARGET2')
RelPos = RelativePositions(supervisor, "TARGET")
print("RelPos.get_pos()", RelPos.get_pos())
# ---------------------- Main Loop Velocity Control-------------------------
print("-------------------------------------------------------")
print("Using Velocity Control to move the end-effector in a circle for 10s.")

# Calculating an initial position and orientation defines the starting position of the
# velocity control inverse kinematic calculations.

robot_base = supervisor.getSelf()
robot_rot_base = np.transpose(np.array(robot_base.getOrientation()).reshape(3, 3))
robot_pos_base = np.array(robot_base.getPosition())
print("robot_pos_base", robot_pos_base)
print("robot_rot_base", robot_rot_base)

print("ikResults", ikResults)
t0 = supervisor.getTime()
while supervisor.step(timeStep * IKstepSize) != -1:
    # Defining x y and z velocities with sinus and cosinus in order to draw a circle
    t = supervisor.getTime()
    x_vel = 0.2 * np.sin(t - t0)
    y_vel = 0.2 * np.cos(t - t0)
    z_vel = 0
    # timeInterval * velocity = new position
    timeInterval = timeStep * IKstepSize / 1000
    # Linear velocity in cartesian spece relative to robot base
    xyz_vel = [x_vel, y_vel, z_vel]
    # Angular velocity (roll, pitch, yaw) relative to the current end-effector orientation
    rpy_vel = [0, 0, 0]
    # Calculate inverse kinematics for given timeInterval, linear and angular velocity
    ikResults = ik.get_ik_velctrl(timeInterval, xyz_vel, rpy_vel)
    # Set the motor positions, if we got a a valid result
    if len(ikResults) == 6:
        for i in range(len(ikResults)):
            motors[i].setPosition(ikResults[i])
    # After 10 seconds, abort and move on to the next loop.
    if t > 10:
        break


# ---------------------- Main Loop -------------------------

print("-------------------------------------------------------")
print("Move or rotate the TARGET sphere to move the arm...")
while supervisor.step(IKstepSize * timeStep) != -1:
    # Get the target position relative to the robot base
    target_pos, target_rot = RelPos.get_pos_static()
    # Get a inverse kinematics solution with the desired position and rotation
    # target_pos is the desired position in [x, y, z] RELATIVE to the robot base
    # target_rot is the desired rotation as 3x3 rotation matrix RELATIVE to the robot base
    ikResults = ik.get_ik(target_pos, target_rot, offset=offset)
    # Set the motor positions, if we got a a valid result
    if len(ikResults) == 6:
        for i in range(len(ikResults)):
            motors[i].setPosition(ikResults[i])
