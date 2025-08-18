from pyrep.robots.end_effectors.gripper import Gripper


class BaxterGripper(Gripper):

    def __init__(self, count: int = 0):
        super().__init__(count, 'YuMiGripper',
                         ['YuMiGripper_leftJoint',
                          'YuMiGripper_rightJoint'])
