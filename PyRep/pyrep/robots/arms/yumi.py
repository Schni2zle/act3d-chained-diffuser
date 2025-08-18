from pyrep.robots.arms.arm import Arm


class YumiLeft(Arm):

    def __init__(self, count: int = 0):
        super().__init__(count, 'Yumi_leftArm', 7, base_name='Yumi')


class YumiRight(Arm):

    def __init__(self, count: int = 0):
        super().__init__(count, 'Yumi_rightArm', 7, base_name='Yumi')
