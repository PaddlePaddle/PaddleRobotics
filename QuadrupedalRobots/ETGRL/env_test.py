import os
import numpy as np
import gym
import argparse
import sys
import rlschool
from rlschool.quadrupedal.envs.env_wrappers.MonitorEnv import Param_Dict
from rlschool.quadrupedal.robots import robot_config
from rlschool.quadrupedal.envs.env_builder import SENSOR_MODE
from copy import copy
import pybullet as p
sensor_mode = copy(SENSOR_MODE)

def param2dynamic_dict(params):
    param = copy(params)
    param = np.clip(param,-1,1)
    dynamic_param = {}
    dynamic_param['control_latency'] = np.clip(40+10*param[0],0,80)
    dynamic_param['footfriction'] = np.clip(0.2+10*param[1],0,20)
    dynamic_param['basemass'] = np.clip(1.5+1*param[2],0.5,3)
    dynamic_param['baseinertia'] = np.clip(np.ones(3)+1*param[3:6],np.array([0.1]*3),np.array([3]*3))
    dynamic_param['legmass'] = np.clip(np.ones(3)+1*param[6:9],np.array([0.1]*3),np.array([3]*3))
    dynamic_param['leginertia'] = np.clip(np.ones(12)+1*param[9:21],np.array([0.1]*12),np.array([3]*12))
    dynamic_param['motor_kp'] = np.clip(80*np.ones(12)+40*param[21:33],np.array([20]*12),np.array([200]*12))
    dynamic_param['motor_kd'] = np.clip(np.array([1.,2.,2.]*4)+param[33:45]*np.array([1,2,2]*4),np.array([0]*12),np.array([5]*12))
    if param.shape[0]>45:
        dynamic_param['gravity'] = np.clip(np.array([0,0,-10])+param[45:48]*np.array([2,2,10]),np.array([-5,-5,-20]),np.array([5,5,-4]))
    return dynamic_param

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load",type=str,default="data/origin_ETG/ESStair_origin.npz")
    parser.add_argument("--video",type=int,default=0)
    parser.add_argument("--task",type=str,default="stairstair")
    parser.add_argument("--suffix",type=str,default="exp")
    parser.add_argument("--save",type=int,default=0)
    parser.add_argument("--step_y",type=str,default=0.05)
    args = parser.parse_args()

    dynamic_param = np.load("data/sigma0.5_exp0_dynamic_param9027.npy")
    dynamic_param = param2dynamic_dict(dynamic_param)
    param = copy(Param_Dict)
    env = rlschool.make_env('Quadrupedal',render = True,task = args.task,
                            dynamic_param = dynamic_param, normal = 1, ETG=1,
                            reward_param = param, sensor_mode = sensor_mode,
                            ETG_path = args.load, step_y = args.step_y)
    obs = env.reset()
    t=0
    td = 0
    action_list = []
    for i in range(600):
        action=np.zeros(12)
        obs, reward, done, info = env.step(action,donef=False)
        action_list.append(info["ETG_act"])
        # if done:
        #     break
    if args.save:
        np.save("gait_action_list_ETG_{}.npy".format(args.suffix),action_list)
if __name__ == "__main__":
    main()
