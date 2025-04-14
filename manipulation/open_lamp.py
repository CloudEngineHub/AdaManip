from matplotlib.widgets import EllipseSelector
from manipulation.base_manipulation import BaseManipulation
from envs.base_env import BaseEnv
from manipulation.utils.transform import *
from logging import Logger
import numpy as np
import torch.nn.functional as F
import random
from collections import deque
import pytorch3d.transforms as tf
from pytorch3d.ops import sample_farthest_points
from pytorch3d.vis.plotly_vis import plot_scene
from pytorch3d.structures import Pointclouds
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from dataset.dataset import Experience, Episode_Buffer, obs_wrapper
import ipdb,time,os
import collections

map_ = {'z':1, 'y':2, 'r':3}

class OpenLampManipulation(BaseManipulation) :

    def __init__(self, env : BaseEnv, cfg : dict, logger : Logger) :

        super().__init__(env, cfg, logger)
    '''
    test env
    '''
    def test_env(self, pose, eval=False):
        batch_size = pose.shape[0]
        self.env.reset()
        pose[:, 2] += self.env.gripper_length*2
        for i in range(2):
            for j in range(15):
                self.env.step(pose)
        pose[:, 2] -= self.env.gripper_length + 0.01
        for i in range(2):
            for j in range(15):
                self.env.step(pose)
        self.env.gripper = True
        for i in range(1000000):
            for j in range(15):
                self.env.step(pose)
        
        '''
        two choice
        '''
        rot_quat = torch.tensor([[ 0, 0, 0.1305262, 0.9914449]]*batch_size, device=self.env.device) 
        flag = True
        step_size = 0.01
        if flag:# rotate
            for i in range(10):
                cur_p = self.env.hand_rigid_body_tensor[:, :3]
                cur_q = self.env.hand_rigid_body_tensor[:,3:7]
                pred_p = cur_p
                pred_q = quat_mul(cur_q, rot_quat)
                pred_pose = torch.cat([pred_p, pred_q], dim=-1).float()
                for j in range(15):
                    self.env.step(pred_pose)   
        else:
            for i in range(4):
                print(self.env.one_dof_tensor[:,0])
                cur_p = self.env.hand_rigid_body_tensor[:, :3]
                cur_q = self.env.hand_rigid_body_tensor[:,3:7]
                cur_p[:, 2] = cur_p[:,2] - step_size
                pred_pose = torch.cat([cur_p, cur_q], dim=-1).float()
                for j in range(15):
                    self.env.step(pred_pose)    
       
    '''
    model test
    '''
    def diffusion_evaluate(self, grasp_net, diffusion):
        eps_num = self.cfg["task"]["num_eval_episode"]
        policy = self.cfg["task"]["policy"]
        max_step = self.cfg["task"]["max_step"]
        succ_cnt = 0
        succ_rate = []
        print("eval_eps_{},max_step_{},policy_{}".format(eps_num, max_step, policy))
        for eps in range(eps_num):
            self.env.reset()
            done_flag = [False] * self.env.num_envs
            self.diffusion_eval_grasp(grasp_net)
            hand_pose = self.env.hand_rigid_body_tensor[:,:7]
            self.env.gripper = True
            for i in range(10):
                self.env.step(hand_pose)
            init_actions = self.action_process(hand_pose)
            self.env.actions = init_actions
            ###############manipulation policy################
            obs = self.env.collect_diff_data()
            pcs, env_state = obs_wrapper(obs)
            pcs_deque = collections.deque([pcs] * diffusion.args.obs_horizon, maxlen=diffusion.args.obs_horizon)
            env_state_deque = collections.deque([env_state] * diffusion.args.obs_horizon, maxlen=diffusion.args.obs_horizon)
            step = 0
            action_horizon = 1
            while step < max_step:
                pred_poses = diffusion.infer_action_with_seg(pcs_deque, env_state_deque).detach()
                action = pred_poses[:, :action_horizon, :]
                step += action_horizon

                for act in range(action.shape[1]):
                    quat = self.rotate_6d_to_quat(action[:, act, 3:])
                    pre_action = torch.cat([action[:, act, :3], quat], dim=-1)
                    self.env.get_obj_dof_property_tensor()

                    for env_id in range(self.env.num_envs):
                        if done_flag[env_id]:
                            pre_action[env_id,:] = hand_pose[env_id,:].clone()
                    for j in range(15):
                        self.env.step(pre_action)
                    
                    self.env.actions = action[:, act, :]
                    obs = self.env.collect_diff_data()
                    pcs, env_state = obs_wrapper(obs)

                    pcs_deque.append(pcs)
                    env_state_deque.append(env_state) 
                
                for env_id in range(self.env.num_envs):
                    if not done_flag[env_id]:
                        if self.env.clock_wise[env_id] == 1:
                            # print(torch.abs(self.env.one_dof_tensor[env_id, 0]))
                            if torch.abs(self.env.one_dof_tensor[env_id, 0]) > 0.01:
                                done_flag[env_id] = True
                                succ_cnt += 1
                                print(f"Env {env_id} Succeeded")
                        else:
                            if torch.abs(self.env.two_dof_tensor[env_id, 0]) > torch.abs(self.env.two_flag[env_id]):
                                done_flag[env_id] = True
                                succ_cnt += 1
                                print(f"Env {env_id} Succeeded")
            cur_rate = succ_cnt/self.env.num_envs
            print(f"Eps {eps+1}, current succ rate {cur_rate}")
            succ_rate.append(cur_rate)
            succ_cnt = 0
        print(f"Average Success rate: {np.mean(succ_rate)}")
        print(f"Success rate std: {np.std(succ_rate)}")
        return
    

    '''
    eval grasp net
    '''
    def diffusion_eval_grasp(self, grasp_net):
        obs = self.env.collect_diff_data()
        pcs, env_state = obs_wrapper(obs)
        pcs_deque = collections.deque([pcs] * grasp_net.args.obs_horizon, maxlen=grasp_net.args.obs_horizon)
        env_state_deque = collections.deque([env_state] * grasp_net.args.obs_horizon, maxlen=grasp_net.args.obs_horizon)
        step = 0
        action_horizon = 3
        while step < 6:
            pred_poses = grasp_net.infer_action_with_seg(pcs_deque, env_state_deque).detach()
            action = pred_poses[:, :action_horizon, :]
            step += action_horizon
            
            for act in range(action.shape[1]):
                quat = self.rotate_6d_to_quat(action[:, act, 3:])
                pre_action = torch.cat([action[:, act, :3], quat], dim=-1)
                self.env.get_obj_dof_property_tensor()

                for j in range(10):
                    self.env.step(pre_action)
                # ipdb.set_trace()
                self.env.actions = action[:, act, :]
                obs = self.env.collect_diff_data()
                pcs, env_state = obs_wrapper(obs)

                pcs_deque.append(pcs)
                env_state_deque.append(env_state)
    
    '''
    collect grasp data
    '''
    def collect_grasp_data(self):
        eps_num = self.cfg["task"]["num_episode"]
        demo_buffer = Experience()
        for eps in range(eps_num):
            self.eps_buffer = [Episode_Buffer() for _ in range(self.env.num_envs)]
            print("eps_{}".format(eps+1))
            self.env.reset()
            pre_pose = self.env.adjust_hand_pose.clone() 
            pre_pose[:, 2] += self.env.gripper_length*2
            for i in range(3):
                obs = self.env.collect_diff_data()
                pc, env_state = obs_wrapper(obs)

                for j in range(10):
                    self.env.step(pre_pose)

                gt_action = self.action_process(pre_pose)
                self.env.actions = gt_action.clone()
                for env_id in range(self.env.num_envs):
                    self.eps_buffer[env_id].add(pc[env_id], env_state[env_id], gt_action[env_id])
            pre_pose[:, 2] -= self.env.gripper_length + 0.01
            for i in range(3):
                obs = self.env.collect_diff_data()
                pc, env_state = obs_wrapper(obs)

                for j in range(10):
                    self.env.step(pre_pose)
                
                gt_action = self.action_process(pre_pose)
                self.env.actions = gt_action.clone()
                for env_id in range(self.env.num_envs):
                    self.eps_buffer[env_id].add(pc[env_id], env_state[env_id], gt_action[env_id])

            # update env end flag
            for env_id in range(self.env.num_envs):
                demo_buffer.append(self.eps_buffer[env_id])
            print(f"Episode {eps} Succeeded")

        if self.cfg['env']['collectData']:
            dataset_path = "grasp_lamp" + "_" + str(self.cfg["env"]["asset"]["AssetNum"])+"_eps"+str(self.cfg["task"]["num_episode"])+"_clock"+str(self.cfg["env"]["clockwise"])
            save_dir = './demo_data/'+ dataset_path
            save_path = save_dir + '/demo_data.zip'
            os.makedirs(save_dir, exist_ok=True)
            demo_buffer.save(save_path)
   
    
    '''
    collect data
    '''
    def collect_manip_data(self):
        eps_num = self.cfg["task"]["num_episode"]
        policy = self.cfg["task"]["policy"]
        print(f"policy: {policy}")
        primitive_action_step = 0.01
        rot_quat = torch.tensor([ 0, 0, -0.1305262, 0.9914449], device=self.env.device) 
        s_rot_quat = torch.tensor([ 0, 0, 0.1305262, 0.9914449], device=self.env.device) 
        demo_buffer = Experience()
        hand_pose = self.env.hand_rigid_body_tensor[:,:7]
        max_step = 15 if policy == "adaptive" else 10
        succ_cnt = [0] * self.env.num_envs
        for eps in range(eps_num):
            chose_list = [['z','y','r'] for _ in range(self.env.num_envs)]
            eps_buffer = [Episode_Buffer() for _ in range(self.env.num_envs)]
            done_flag = [False] * self.env.num_envs
            print("eps_{}".format(eps+1))
            self.env.reset()
            pre_pose = self.env.adjust_hand_pose.clone() 
            pre_pose[:, 2] += self.env.gripper_length*2
            for i in range(3):
                for j in range(10):
                    self.env.step(pre_pose)

            pre_pose[:, 2] -= self.env.gripper_length + 0.01
            for i in range(3):
                for j in range(10):
                    self.env.step(pre_pose)

            self.env.gripper = True
            for i in range(10):
                self.env.step(hand_pose)

            init_actions = self.action_process(hand_pose)
            self.env.actions = init_actions
            ####################start collect manipulation data###################
            for t in range(max_step):
                cur_p = hand_pose[:, :3]
                cur_q = hand_pose[:,3:7]
                pre_p = cur_p.clone()
                pre_q = cur_q.clone()

                for i in range(self.env.num_envs):
                    if policy == "succ":
                        res = self.succ_policy(i)
                    elif policy == "adaptive":
                        res = self.ada_policy(i, t, chose_list[i])
                    else:
                        raise NotImplementedError
                    # if not done_flag[i]:
                    #     print(res)
                    if res == "z":
                        pre_p[i,2] -= primitive_action_step
                    elif res == "r":
                        pre_q[i] = quat_mul(cur_q[i], s_rot_quat)
                    elif res == "y":
                        pre_q[i] = quat_mul(cur_q[i], rot_quat)
                
                pred_pose = torch.cat([pre_p, pre_q], dim=-1).float()
                gt_pose = self.action_process(pred_pose)

                for env_id in range(self.env.num_envs):
                    if not done_flag[env_id]:
                        obs = self.env.collect_single_diff_data(env_id)
                        pc, env_state = obs_wrapper(obs)
                        eps_buffer[env_id].add(pc, env_state, gt_pose[env_id])

                for j in range(15):
                    self.env.step(pred_pose)
                
                self.env.actions = gt_pose

                # update env end flag
                for env_id in range(self.env.num_envs):
                    if not done_flag[env_id]:
                        if self.env.clock_wise[env_id] == 1:
                            # print(torch.abs(self.env.one_dof_tensor[env_id, 0]))
                            if torch.abs(self.env.one_dof_tensor[env_id, 0]) > 0.007:
                                demo_buffer.append(eps_buffer[env_id])
                                done_flag[env_id] = True
                                succ_cnt[env_id] += 1
                                print(f"Env {env_id} Succeeded")
                        else:
                            if torch.abs(self.env.two_dof_tensor[env_id, 0]) > torch.abs(self.env.two_flag[env_id]):
                                demo_buffer.append(eps_buffer[env_id])
                                done_flag[env_id] = True
                                succ_cnt[env_id] += 1
                                print(f"Env {env_id} Succeeded")
            print(succ_cnt)
        if self.cfg['env']['collectData']:
            dataset_path = "manip_lamp"+'_'+self.cfg["task"]["policy"] + "_" + str(self.cfg["env"]["asset"]["AssetNum"])+"_eps"+str(self.cfg["task"]["num_episode"])+"_clock"+str(self.cfg["env"]["clockwise"])
            save_dir = './demo_data/'+ dataset_path
            save_path = save_dir + '/demo_data.zip'
            os.makedirs(save_dir, exist_ok=True)
            demo_buffer.save(save_path)
    
    def succ_policy(self, env_id):
        clock_wise = self.env.clock_wise[env_id]
        if clock_wise == 1:
            return 'z'
        elif clock_wise == 2:
            return 'y'
        elif clock_wise == 3:
            return 'r'
    
    def ada_policy(self, env_id, t, cho_list):
        clock_wise = self.env.clock_wise[env_id]
        prob = np.random.rand()*3
        if t == 0:
            if prob < 1:
                self.env.action_chosen[env_id, t] = 'z'
                cho_list.remove('z')
                return 'z'
            elif prob < 2:
                self.env.action_chosen[env_id, t] = 'y'
                cho_list.remove('y')
                return 'y'
            else:
                self.env.action_chosen[env_id, t] = 'r'
                cho_list.remove('r')
                return 'r'
        
        if map_[self.env.action_chosen[env_id, t - 1]] == clock_wise:
            action = self.env.action_chosen[env_id, t - 1]
            self.env.action_chosen[env_id, t] = action
            return action
        else:
            if len(cho_list) == 2:
                s_prob = np.random.rand()
                if s_prob < 11/20:
                    action = cho_list[0]
                    cho_list.remove(action)
                    self.env.action_chosen[env_id, t] = action
                    return action
                else:
                    action = cho_list[1]
                    cho_list.remove(action)
                    self.env.action_chosen[env_id, t] = action
                    return action
            else:
                action = cho_list[0]
                cho_list.remove(action)
                self.env.action_chosen[env_id, t] = action
                return action
            
    def action_process(self, pose):
        quat = pose[:,3:7]
        rotate_matix = tf.quaternion_to_matrix(quat)
        rotate_6d = tf.matrix_to_rotation_6d(rotate_matix)
        return torch.cat([pose[:,:3], rotate_6d], dim=-1)

    def rotate_6d_to_quat(self, rotate_6d):
        rotate_matix = tf.rotation_6d_to_matrix(rotate_6d)
        quat = tf.matrix_to_quaternion(rotate_matix)
        return quat