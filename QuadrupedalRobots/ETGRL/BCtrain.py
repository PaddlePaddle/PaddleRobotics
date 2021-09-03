#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import os
import torch
import numpy as np
import gym
import argparse
from parl.utils import logger, summary
from alg.BCreplay_buffer import BCReplayMemory
from model.mujoco_model import MujocoModel
from model.mujoco_agent import MujocoAgent
from alg.sac import SAC
from alg.BC import BC
from rlschool.quadrupedal.envs.env_wrappers.MonitorEnv import Param_Dict,Random_Param_Dict
from rlschool.quadrupedal.robots import robot_config
from rlschool.quadrupedal.envs.env_builder import SENSOR_MODE
import rlschool
from copy import copy
import pybullet as p
import cv2
import time

WARMUP_STEPS = 200
EVAL_EVERY_STEPS = 1e4
EVAL_EPISODES = 1
MEMORY_SIZE = int(1e7)
TRAIN_PER_STEPS = 1024
TRAIN_PER_TIME = 10
BATCH_SIZE = 1024
GAMMA = 0.99
TAU = 0.005
ALPHA = 0.2  # determines the relative importance of entropy term against the reward
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4

param = copy(Param_Dict)
random_param = copy(Random_Param_Dict)
mode_map ={"pose":robot_config.MotorControlMode.POSITION,
            "torque":robot_config.MotorControlMode.TORQUE,
            "traj":robot_config.MotorControlMode.POSITION,}

def obs2noise(obs):
    obs_noise = copy(obs)
    obs_noise[7:10] += np.random.normal(0,6e-2,size=3)/0.1
    obs_noise[10:13] += np.random.normal(0,1e-1,size=3)/0.5
    obs_noise[13:25] += np.random.normal(0,1e-2,size=12)/0.1
    obs_noise[25:37] += np.random.normal(0,0.5,size=12)
    return obs_noise

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

def cal_agent_obs(obs):
    obs_now = copy(obs)
    if args.sensor_noise:
        obs_now = obs2noise(obs_now)
    return obs_now[3:]

def cal_ref_obs(obs):
    return obs

# Run episode for training
def run_train_episode(agent, env, rpm,max_step,action_bound,ref_agent=None,total_steps=0):
    action_dim = env.action_space.shape[0]
    obs,info = env.reset(x_noise=args.x_noise)
    done = False
    episode_reward, episode_steps = 0, 0
    infos = {}
    success_num = 0
    actor_loss_list = []
    critic_loss_list = []
    train_flag = 0
    while not done:
        agent_obs = cal_agent_obs(obs)
        ref_obs = cal_ref_obs(obs)
        episode_steps += 1
        # Select action randomly or according to policy
        if rpm.size() < WARMUP_STEPS:
            action = np.random.uniform(-1, 1, size=action_dim)
        else:
            action = agent.sample(agent_obs)
        new_action = copy(action)
        # Perform action
        next_obs, reward, done, info = env.step(new_action*action_bound,donef=(episode_steps>max_step))
        terminal = float(done) if episode_steps < 2000 else 0
        terminal = 1. - terminal
        for key in Param_Dict.keys():
            if key in info.keys():
                if key not in infos.keys():
                    infos[key] = info[key]
                else:
                    infos[key] += info[key]
        if info["velx"]>=0.3:
            success_num +=1
        # Store data in replay memory
        rpm.append(agent_obs, ref_obs)
        obs = next_obs
        episode_reward += reward
        if (rpm.size() >= WARMUP_STEPS) and ((total_steps+episode_steps)%TRAIN_PER_STEPS == 0):
            train_flag = 1
        if episode_steps > max_step:
            break
        # Train agent after collecting sufficient data
    if train_flag:
        for t in range(TRAIN_PER_TIME):
            random_list = np.arange(rpm.size())
            np.random.shuffle(random_list)
            for j in range(0,rpm.size()-BATCH_SIZE,BATCH_SIZE):
                batch_idx = random_list[j:j+BATCH_SIZE]
                batch_obs, batch_ref_obs= rpm.sample_batch_by_index(
                    batch_idx)
                critic_loss,actor_loss = agent.BClearn(batch_obs, batch_ref_obs,ref_agent)
                actor_loss_list.append(actor_loss)
                critic_loss_list.append(critic_loss)
    if len(actor_loss_list)>0:
        infos["actor_loss"] = np.mean(actor_loss_list)
        infos["critic_loss"] = np.mean(critic_loss_list)
    infos["success_rate"] = success_num/episode_steps
    return episode_reward, episode_steps,infos

# Runs policy for 5 episodes by default and returns average reward
# A fixed seed is used for the eval environment
def run_evaluate_episodes(agent, env,max_step,action_bound,ref_agent=None):
    avg_reward = 0.
    infos = {}
    steps_all = 0
    obs,info = env.reset(x_noise=args.x_noise)
    done = False
    steps = 0
    while not done:
        steps +=1
        if ref_agent is None:
            action = agent.predict(cal_agent_obs(obs))
        else:
            action = ref_agent.predict(cal_ref_obs(obs))
        new_action = action
        obs, reward, done, info = env.step(new_action*action_bound,donef=(steps>max_step))
        if args.eval == 1:
            img=p.getCameraImage(640, 480, renderer=p.ER_BULLET_HARDWARE_OPENGL)[2]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB ) 
            cv2.imwrite("img/img{}.jpg".format(steps),img)
        avg_reward += reward
        for key in Param_Dict.keys():
            if key in info.keys():
                if key not in infos.keys():
                    infos[key] = info[key]
                else:
                    infos[key] += info[key]
        if steps > max_step:
            break
    steps_all += steps
    return avg_reward,steps_all,infos

def run_random_eval(agent, env,max_step,act_bound,total_steps,ref_agent,eval_t=EVAL_EPISODES):
    rewards = []
    infos_all = {}
    steps_all = []
    for i in range(eval_t):
        reward,steps,infos = run_evaluate_episodes(agent,env,max_step,act_bound,ref_agent=None)
        ref_reward,ref_steps,ref_infos = run_evaluate_episodes(agent,env,max_step,act_bound,ref_agent=ref_agent)
        ratio = reward/ref_reward
        infos["ref_ratio"] = ratio
        for key in infos.keys():
            if infos[key]!=0:
                if key not in infos_all.keys():
                    infos_all[key] = 0
                infos_all[key] += infos[key]/eval_t
        rewards.append(reward)
        steps_all.append(steps)
    summary.add_scalar('eval/reward_min', np.min(rewards),
                                        total_steps)
    summary.add_scalar('eval/reward_std', np.std(rewards),
                                    total_steps)
    return np.mean(rewards),np.mean(steps_all),infos_all


def main():
    random_param['random_dynamics'] = args.random_dynamic
    random_param['random_force'] = args.random_force
    param['torso'] = args.torso
    param['feet'] = args.feet
    param['up'] = args.up
    param['tau'] = args.tau
    param['stand'] = args.stand
    sensor_mode = copy(SENSOR_MODE)
    sensor_mode['dis'] = args.sensor_dis
    sensor_mode['motor'] = args.sensor_motor
    sensor_mode["imu"] = args.sensor_imu
    sensor_mode["contact"] = args.sensor_contact
    sensor_mode["ETG"] = args.sensor_ETG
    sensor_mode["footpose"] = args.sensor_footpose
    sensor_mode["ETG_obs"] = args.sensor_ETG_obs
    sensor_mode["dynamic_vec"] = args.sensor_dynamic
    sensor_mode["force_vec"] = args.sensor_exforce
    rnn_config = {}
    rnn_config["time_steps"] = args.timesteps
    rnn_config["time_interval"] = args.timeinterval
    rnn_config["mode"] = args.RNN_mode
    sensor_mode["RNN"] = rnn_config
    render = True if (args.eval or args.render )else False
    mode = mode_map[args.act_mode]
    dynamic_param = np.load("data/sigma0.5_exp0_dynamic_param9027.npy")
    dynamic_param = param2dynamic_dict(dynamic_param)
    env =  rlschool.make_env('Quadrupedal',task=args.task_mode,motor_control_mode=mode,render=render,sensor_mode=sensor_mode,
                        normal=args.normal,dynamic_param=dynamic_param,reward_param=param,
                        ETG=args.ETG,ETG_T=args.ETG_T,reward_p=args.reward_p,ETG_path=args.ETG_path,random_param=random_param,
                        ETG_H = args.ETG_H, vel_d = args.vel_d,step_y=args.step_y,
                        enable_action_filter=args.enable_action_filter)
    e_step = args.e_step
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print('obs_dim:',obs_dim)
    act_bound_now = args.act_bound
    if args.act_mode == "pose":
        act_bound = np.array([0.1,0.7,0.7]*4)
    elif args.act_mode == "torque":
        act_bound = np.array([10]*12)
    elif args.act_mode == "traj":
        act_bound = np.array([act_bound_now,act_bound_now,act_bound_now]*4)
    if args.agent_mode == "stack":
        agent_obs_dim = (obs_dim-3)*6
    else:
        agent_obs_dim = obs_dim-3
    print("agent_obs_dim:",agent_obs_dim)
    # Initialize model, algorithm, agent, replay_memory
    ref_model = MujocoModel(obs_dim, action_dim)
    rpm = BCReplayMemory(
        max_size=MEMORY_SIZE, obs_dim=agent_obs_dim, 
        act_dim=action_dim,ref_obs_dim=obs_dim)
    ref_algorithm = SAC(
        ref_model,
        gamma=GAMMA,
        tau=TAU,
        alpha=ALPHA,
        actor_lr=ACTOR_LR,
        critic_lr=CRITIC_LR)
    ref_agent = MujocoAgent(ref_algorithm)
    ref_agent.restore(args.ref_agent)
    model = MujocoModel(agent_obs_dim,action_dim)
    algorithm = BC(
        model,
        actor_lr=ACTOR_LR,
        critic_lr=CRITIC_LR)
    agent = MujocoAgent(algorithm,alg_mode="BC")
    if len(args.load)>0:
        agent.restore(args.load)
    if not args.eval:
        outdir = os.path.join(args.outdir,args.suffix)
        if not os.path.exists(args.outdir):
            os.makedirs(args.outdir)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        logger.set_dir(outdir)
        logger.info('args:{}'.format(args))
        total_steps = 0 
        test_flag = 0
        t_steps = 0
        while total_steps < args.max_steps:
            # Train episode
            episode_reward, episode_step,info = run_train_episode(agent, env, rpm,e_step,act_bound,ref_agent=ref_agent,total_steps=total_steps)
            total_steps += episode_step
            t_steps += episode_step
            summary.add_scalar('train/episode_reward', episode_reward,
                                total_steps)
            summary.add_scalar('train/episode_step', episode_step,
                                total_steps)
            for key in info.keys():
                if info[key] != 0:
                    summary.add_scalar('train/episode_{}'.format(key),info[key],total_steps)
                    summary.add_scalar('train/mean_{}'.format(key),info[key]/episode_step,total_steps)   
            logger.info('Total Steps: {} Reward: {} '.format(
                total_steps, episode_reward))

            # Evaluate episode
            if (total_steps + 1) // EVAL_EVERY_STEPS >= test_flag:
                while (total_steps + 1) // EVAL_EVERY_STEPS >= test_flag:
                    test_flag += 1
                    avg_reward,avg_step,info = run_random_eval(agent, env,800,act_bound,total_steps,ref_agent)
                    summary.add_scalar('eval/episode_reward', avg_reward,
                                        total_steps)
                    summary.add_scalar('eval/episode_step', avg_step,
                                        total_steps)   
                    for key in info.keys():
                        if info[key] != 0:
                            summary.add_scalar('eval/episode_{}'.format(key),info[key],total_steps)
                            summary.add_scalar('eval/mean_{}'.format(key),info[key]/avg_step,total_steps) 
                    logger.info('Evaluation over: {} episodes, Reward: {} Steps: {}'.format(
                        EVAL_EPISODES, avg_reward,avg_step))
                if e_step<600:
                        e_step +=50
                path = os.path.join(outdir,'itr_{:d}.pt'.format(int(total_steps)))
                agent.save(path)
    elif args.eval == 1:
        outdir = os.path.join(args.load[:-3],args.terrain+"height{}_wid{}_slope{}".format(args.stepheight,args.stepwidth,args.slope))
        if not os.path.exists(args.load[:-3]):
            os.makedirs(args.load[:-3])
        avg_reward,avg_step,info = run_evaluate_episodes(agent, env,600,act_bound,ref_agent=None)
        
        os.system("ffmpeg -r 38 -i img/img%01d.jpg -vcodec mpeg4 -vb 40M -y {}.mp4".format(outdir))
        os.system("rm -rf img/*")
        logger.info('Evaluation over: {} episodes, Reward: {} Steps: {} StepHeight: {}'.format(
                    EVAL_EPISODES, avg_reward,avg_step,stepheight))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir",type=str,default="BCtrain_log")
    parser.add_argument("--max_steps",type=int,default=1e6)
    parser.add_argument("--epsilon",type=float,default=0.4)
    parser.add_argument("--gamma",type=float,default=0.95)
    parser.add_argument("--load", type=str, default="", help="Directory to load agent from.")
    parser.add_argument("--eval", type=int, default=0, help="Evaluate or not")
    parser.add_argument("--suffix",type=str,default="exp0")
    parser.add_argument("--task_mode",type=str,default="stairstair")
    parser.add_argument("--step_y",type=float,default=0.05)
    parser.add_argument("--random_dynamic",type=int,default=0)
    parser.add_argument("--random_force",type=int,default=0)
    parser.add_argument("--render", type=int, default=0, help="render or not")
    parser.add_argument("--normal",type=int,default=1)
    parser.add_argument("--vel_d",type=float,default=0.6)
    parser.add_argument("--ETG",type=int,default=1)
    parser.add_argument("--ETG_T",type=float,default=0.5)
    parser.add_argument("--reward_p",type=float,default=1)
    parser.add_argument("--ETG_T2",type=float,default=0.5)
    parser.add_argument("--e_step",type=int,default=400)
    parser.add_argument("--act_mode",type=str,default="traj")
    parser.add_argument("--ref_agent",type=str,default="data/model/StairStair_3_itr_960231.pt")
    parser.add_argument("--ETG_path",type=str,default="data/model/StairStair_3_itr_960231.npz")
    parser.add_argument("--ETG_H",type=int,default=20)
    parser.add_argument("--stand",type=float,default=0)
    parser.add_argument("--torso",type=float,default=1)
    parser.add_argument("--up",type=float,default=0.1)
    parser.add_argument("--tau",type=float,default=0.1)
    parser.add_argument("--feet",type=float,default=0.1)
    parser.add_argument("--cl_method",type=str,default="None")
    parser.add_argument("--act_bound",type=float,default=0.3)
    parser.add_argument("--sensor_dis",type=int,default=1)
    parser.add_argument("--sensor_motor",type=int,default=1)
    parser.add_argument("--sensor_imu",type=int,default=1)
    parser.add_argument("--sensor_contact",type=int,default=1)
    parser.add_argument("--sensor_ETG",type=int,default=1)
    parser.add_argument("--sensor_footpose",type=int,default=0)
    parser.add_argument("--sensor_ETG_obs",type=int,default=0)
    parser.add_argument("--sensor_dynamic",type=int,default=0)
    parser.add_argument("--sensor_exforce",type=int,default=0)
    parser.add_argument("--sensor_noise",type=int,default=1)
    parser.add_argument("--timesteps",type=int,default=5)
    parser.add_argument("--timeinterval",type=int,default=1)
    parser.add_argument("--RNN_mode",type=str,default="None")
    parser.add_argument("--agent_mode",type=str,default="None")
    parser.add_argument("--enable_action_filter",type=int,default=0)
    parser.add_argument("--x_noise",type=int,default=0)
    args = parser.parse_args()

    main()
