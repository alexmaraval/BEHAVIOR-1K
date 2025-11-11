import torch
from omnigibson.reward_functions.reward_function_base import BaseRewardFunction
import omnigibson.utils.transform_utils as T
from omnigibson.object_states import Pose, Joint

class StandUprightReward(BaseRewardFunction):
    def __init__(self, robot_idn=0, coeff=1.0):
        self._robot_idn = robot_idn
        self.coeff = float(coeff)
        super().__init__()

    def _step(self, task, env, action):
        current_joint_state = env.robots[self._robot_idn].states[Joint].get_value()
        original_joint_state = torch.zeros(28) * torch.pi / 180
        torso_diff = (current_joint_state - original_joint_state)[6:10].norm()
        r = (1 + torch.tanh(-torso_diff)) * self.coeff
        return float(r.item()), {"upright": float(r.item())}
        # pos, quat = env.robots[self._robot_idn].states[Pose].get_value()
        # R = T.quat2mat(quat)
        # up = R[:, 2]        # robot's local +Z in world
        # c = torch.clamp(up[2], -1.0, 1.0) # dot(up, world_z) == R[2,2]
        # r = pos[-1] * self.coeff
        # return float(r.item()), {"upright_cos": float(c.item())}