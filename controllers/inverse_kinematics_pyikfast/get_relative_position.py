import numpy as np


class RelativePositions():
    def __init__(self, Supervisor,  DEF_target, DEF_base=None):
        self.robot = Supervisor
        self.target = self.robot.getFromDef(DEF_target)
        if DEF_base is None:
            self.base = self.robot.getSelf()
        else:
            self.base = self.robot.getFromDef(DEF_base)
        self.rot_base = np.transpose(
            np.array(self.base.getOrientation()).reshape(3, 3))
        self.pos_base = np.array(self.base.getPosition())

    def get_pos(self, offset_target_frame=[0, 0, 0]):
        """Returns the translation vector and 3x3 Rotation matrix from the BASE to the TARGET
        'offset_target_frame' is an offset in the coordinate system of the TARGET.
        Example: We want the position of the top of a 0.4m high cylinder, not its center. 
        To do this, we can specify 
        offset_target_frame=[0, 0.2, 0]
        """
        # Get the transposed rotation matrix of the base, so we can calculate poses of
        # everything relative to it.
        # Get orientation of the Node we want as our new reference frame and turn it into
        # a numpy array. Returns 1-dim list of len=9.
        rot_base = np.array(self.base.getOrientation())
        # reshape into a 3x3 rotation matrix
        rot_base = rot_base.reshape(3, 3)
        # Transpose the matrix, because we need world relative to the base, not the
        # base relative to world.
        rot_base = np.transpose(rot_base)
        # Get the translation between the base and the world (basically where the origin
        # of our new relative frame is).
        # No need to use the reverse vector, as we will subtract instead of add it later.
        pos_base = np.array(self.base.getPosition())

        # target position relative to world.
        target_pos_world = np.array(self.target.getPosition())
        # target rotation relative to world
        target_rot_world = np.array(self.target.getOrientation()).reshape(3, 3)
        # offset calculation
        target_pos_world = target_pos_world + \
            np.dot(target_rot_world, offset_target_frame)
        # Calculate the relative translation between the target and the base.
        target_pos_world = np.subtract(target_pos_world, pos_base)
        # Matrix multiplication with rotation matrix: target posistion relative to base.
        target_pos_base = np.dot(rot_base, target_pos_world)
        # Calculate the orientation of the target, relative to the base, all in one line.
        target_rot_base = np.dot(rot_base, target_rot_world)
        return target_pos_base, target_rot_base

    def get_pos_static(self, offset_target_frame=[0, 0, 0]):
        '''get relative position and orientation, for a static base. This method is 
        faster, as it doesnt have to make the extra calculations for the base every time.'''
        # target position relative to world.
        target_pos_world = np.array(self.target.getPosition())
        # target rotation relative to world
        target_rot_world = np.array(self.target.getOrientation()).reshape(3, 3)
        # offset calculation
        target_pos_world = target_pos_world + \
            np.dot(target_rot_world, offset_target_frame)
        # Matrix multiplication with rotation matrix: target posistion relative to base.
        target_pos_base = np.dot(self.rot_base, target_pos_world)

        # Calculate the orientation of the target, relative to the base, all in one line.
        target_rot_base = np.dot(self.rot_base, target_rot_world)
        return target_pos_base, target_rot_base
