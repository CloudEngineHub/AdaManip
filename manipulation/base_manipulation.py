from abc import abstractclassmethod
from envs.base_env import BaseEnv
from logging import Logger
import pytorch3d.transforms as tf
import torch

class BaseManipulation :

    def __init__(self, env : BaseEnv, cfg : dict, logger : Logger) :

        self.env = env
        self.cfg = cfg
        self.logger = logger

    @abstractclassmethod
    def collect_data(self, obs, eval=False) :

        pass

    def action_process(self, pose):
        quat_isaac = pose[:,3:7]
        quat_p3d = torch.cat([quat_isaac[:,3:], quat_isaac[:,:3]], dim=-1)
        rotate_matix = tf.quaternion_to_matrix(quat_p3d)
        rotate_6d = tf.matrix_to_rotation_6d(rotate_matix)
        return torch.cat([pose[:,:3], rotate_6d], dim=-1)

    def rotate_6d_to_quat(self, rotate_6d):
        rotate_matix = tf.rotation_6d_to_matrix(rotate_6d)
        quat_p3d = tf.matrix_to_quaternion(rotate_matix)
        quat_isaac = torch.cat([quat_p3d[:,1:], quat_p3d[:,:1]], dim=-1)
        return quat_isaac