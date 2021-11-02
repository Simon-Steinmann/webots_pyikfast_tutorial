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

import numpy as np
from transforms3d import euler, quaternions


class inverseKinematics():
    def __init__(self, pyikfast_module, name='Irb4600-40', minPositions=[-2 * np.pi] * 6, maxPositions=[2 * np.pi] * 6):
        # Bounds for joint motors. Only valid IK solutions will be returned
        self.boundsMin = np.array(minPositions[:6])
        self.boundsMax = np.array(maxPositions[:6])
        # This dict specifies the angle of joint_2, where the arm stands straight up
        offset_dict = {
            'UR3e': -np.pi / 2,
            'UR5e': -np.pi / 2,
            'UR10e': -np.pi / 2,
            'Irb4600-40': 0,
            'P-Rob3': 0,
            'IprHd6ms90': -np.pi / 2,
            'Puma560': -np.pi / 2}
        # Set the joint2_offset. Needed for picking the best ik-solution
        try:
            self.joint2_offset = offset_dict[name]
        except:
            print('WARNING: ', name,
                  ' - needs to be configured in the ik_module! Please adjust the offset_dict!')
            self.joint2_offset = 0

        # Initialize kinematics for our robot arm
        self.ikfast = pyikfast_module
        ## self.n_joints = self.ikfast.get_dof()
        self.n_joints = 6
        print('Number of DOF: ', self.n_joints)
        self.old_ikResults = np.zeros(self.n_joints)
        self.last_pos, self.last_rot = self.get_fk(self.old_ikResults)
        # Incremental rotation matrix, rotating each axis by 0.005 radians.
        # If we encounter a singularity, the target rotation matrix will
        # be multiplied with this increment rotation matrix.
        self.increment = self.rotate_matrix(np.identity(3), [0.005] * 3)

        # Velocity control variables
        # last rotation in quaternion notation
        self.last_quat = quaternions.mat2quat(self.last_rot)
        self.velctrl_active = False
        self.result_last_valid = self.old_ikResults
        # For advanced velocity control. We can set a bounding box for min and max
        # x y z values. The endeffector will only move inside this space.
        self.valid_space = np.array(
            [- np.full((3), np.inf), np.full((3), np.inf)])

    def get_fk(self, joint_positions):
        ''' Calculate the position and rotation of the endeffektor from joint angles'''
        pos, rot = self.ikfast.forward(joint_positions.tolist())
        return np.array(pos), np.array(rot).reshape(3, 3)

    def get_ik_velctrl(self, timeInterval, xyz_vel, rpy_vel=[0, 0, 0]):
        '''
        Move the end-effecor using cartesian velocities. These are relative to the robot base.
        We calculate joint positions by incrementing the last valid position
        with the desired linear and angular velocities. We dont use the actual
        end-effector position, because this would accumulate errors. Instead we have a invisible
        target point, which we move with each loop by the given velocities. Then we 
        calculate an ik solution for this new target point.
        '''
        # If we are calling this function the first time, we have to set some variables.
        # This allows us to position our robot with get_ik first, and then use velctrl.
        if not self.velctrl_active:
            self.last_quat = quaternions.mat2quat(self.last_rot)
            self.result_last_valid = self.old_ikResults
            self.velctrl_active = True
        # Calculate the xyz and rpy deltas relative to velocities and timestep
        xyz_delta = np.array(xyz_vel, dtype='float') * timeInterval
        rpy_delta = np.array(rpy_vel, dtype='float') * timeInterval
        # New position
        pos = self.last_pos + xyz_delta
        # New rotation. q is quaternion, R is 3x3 rotation matrix
        q, R = self.rotate_quat(self.last_quat, rpy_delta)
        # Set old_ikResutls to the last valid.
        self.old_ikResults = self.result_last_valid
        # Calculate inverse kinematics
        pos = self.xyz_withinBounds(pos)
        ikResults = self.get_ik(pos, R)
        # Check if we got at least one solution
        if len(ikResults) > 0:
            # Calculate the sum of how much we moved joint 1-5
            ikResult_sum = np.sum(
                np.abs(ikResults[:5] - self.result_last_valid[:5]))
            p6Delta = self.result_last_valid[5] - ikResults[5]
            index = np.argmin(
                np.abs(np.array([p6Delta - 2 * np.pi, p6Delta, p6Delta + 2 * np.pi])))
            ikResults[5] -= (index - 1) * 2 * np.pi
            #print(p6Delta, ikResults)
            # If this value is too big, it means that we require a large movement and the
            # end-effector would jump.
            if ikResult_sum < 50 * timeInterval:
                # Set our result as the last valid
                self.result_last_valid = ikResults
                # Update our last position and quaternion (orientation)
                self.last_pos = pos
                self.last_quat = q
                # Return the results
                return ikResults
        # Return empty results if no valid is found
        return []

    def get_ik(self, target_pos, target_rot=np.identity(3), offset=[0, 0, 0]):
        '''Get a Inverse Kinematics solution for a given  target position and rotation.
        target_pos is the xyz translation vector between the robot base and desired position.
        target_rot is the 3x3 rotation matrix between the robot base and desired orientation.
        offset is the xyz offset relative to the gripper. This can be useful when using a
        gripper or other tool attached to the robot.'''
        # fill the 3x4 ee_pose matrix with our target rotation and position
        target_pos = target_pos - np.dot(target_rot, offset)

        self.last_rot = target_rot
        self.last_pos = target_pos
        # Inverse kinematics: get joint angles from end effector pose
        #ikResults = np.array(self.ikfast.get_ik(target_pos.tolist(), target_rot.tolist()))
        ikResults = np.array(self.ikfast.inverse(
            target_pos.tolist(), target_rot.reshape(9).tolist()))
        # If we don't get a solution, we try to slightly alter the rotation, to avoid a singularity
        if len(ikResults) == 0:
            for i in range(10):
                target_rot = np.dot(target_rot, self.increment)
                ikResults = np.array(self.ikfast.inverse(
                    target_pos.tolist(), target_rot.reshape(9).tolist()))
                if len(ikResults) != 0:
                    break
        try:
            # Shifts the angles, so the range is [-pi, pi] instead of [0, 2*pi]
            ikResults = np.where(
                ikResults > np.pi, ikResults - 2 * np.pi, ikResults)
            # Check if the ikResults are within the motor limits
            self.withinBounds(ikResults)
            # Remove any results that are outside of joint limits
            ikResults = ikResults[self.mask, :]
            # We want to select the best solution. To do this, we want the 3rd joint to be as high as possible.
            height_joint3 = np.cos(ikResults[:, 1] - self.joint2_offset) * 5
            # We want the least rotation in the first joint (avoid 180 deg turns). We multiply by 2, as this value
            # is smaller than pi, while joint45_delta can be up to 2*pi
            joint1_delta = np.multiply(
                np.abs(np.subtract(ikResults[:, 0], self.old_ikResults[0])), 2)
            # We want the least rotation in joint 4 and 5 (wrist flipping around)
            joint45_delta = np.abs(np.subtract(
                ikResults[:, 3:5], self.old_ikResults[3:5])).sum(axis=1)
            # Pick the solution with the maximum value for height_joint3, but removes solutions, where the
            # arm has to rotate 180 deg or has to flip its wrist.
            self.ikResult_var = height_joint3 - (joint1_delta + joint45_delta)
            index = np.argmax(self.ikResult_var)
            # Store the ikResult we are returning, so we can use it in the next calculation
            self.old_ikResults = ikResults[index]
            return ikResults[index]
        except:
            print('No solution found for point {}'.format(target_pos))
            return []

    def withinBounds(self, results):
        self.mask = np.array(np.zeros(len(results)), dtype=bool)
        for i in range(len(results)):
            self.mask[i] = (all(results[i] > self.boundsMin)
                            and all(results[i] < self.boundsMax))

    def xyz_withinBounds(self, pos):
        return np.minimum(np.maximum(pos, self.valid_space[0]), self.valid_space[1])

    def rotate_matrix(self, R, rpy):
        q1 = quaternions.mat2quat(R)
        q2 = euler.euler2quat(rpy[0], rpy[1], rpy[2])
        q_new = quaternions.qmult(q1, q2)
        return quaternions.quat2mat(q_new)

    def rotate_quat(self, q1, rpy):
        q2 = euler.euler2quat(rpy[0], rpy[1], rpy[2])
        q_new = quaternions.qmult(q1, q2)
        return q_new, quaternions.quat2mat(q_new)
