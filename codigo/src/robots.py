# -*- coding: utf-8 -*-
from obs import RobotCenteredObservation
from luxai_s2.env import EnvConfig
import numpy as np


def invert_dict(d):
    return {v: k for k, v in d.items()}


_0 = {
    0: 'move',
    1: 'transfer _3 amount of _RESOURCE',
    2: 'pickup _3 amount of _RESOURCE',
    3: 'dig',
    4: 'self-destruct',
    5: 'recharge X'
}
_TYPE = invert_dict(_0)
_1 = {0: 'center', 1: 'up', 2: 'right', 3: 'down', 4: 'left'}
_DIRECTION = invert_dict(_1)
_2 = {0: 'ice', 1: 'ore', 2: 'water', 3: 'metal', 4: 'power'}
_RESOURCE = invert_dict(_2)
_DEFAULT_RESOURCE = 0
_DEFAULT_AMOUNT = 1
_DEFAULT_REPEAT = 0
_DEFAULT_N = 1
"""
_3 = amount
_4 = repeat
_5 = n
"""


class Enacter:
    def __init__(self, robot_obs: RobotCenteredObservation,
                 env_cfg: EnvConfig):
        self.obs = robot_obs
        self.conf = env_cfg

    def move_right(self, finite: int = 1):
        assert 0 < finite < 10000
        return np.array([
            _TYPE['move'], _DIRECTION['right'], _DEFAULT_RESOURCE,
            _DEFAULT_AMOUNT, _DEFAULT_REPEAT, finite
        ])


if __name__ == "__main__":
    pass
