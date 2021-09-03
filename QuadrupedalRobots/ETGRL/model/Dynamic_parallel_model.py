import numpy as np
import gym
import sys
import os
from alg.es import SimpleGA, PEPG, OpenES,SimpleES
import time
from rlschool.quadrupedal.envs.env_wrappers.MonitorEnv import Param_Dict,Random_Param_Dict
from parl.utils import logger, summary
import parl
from copy import copy
base_foot = np.array([ 0.17,-0.135,-0.2,0.17,0.13,-0.2,\
                        -0.195,-0.135,-0.2,-0.195,0.13,-0.2])

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
    dynamic_param['gravity'] = np.clip(np.array([0,0,-10])+param[45:48]*np.array([2,2,10]),np.array([-5,-5,-20]),np.array([5,5,-4]))
    return dynamic_param

def loss_func(drpy,motor,mean_dict,key):

    diff_motor = motor-mean_dict[key+"_motor_mean"]
    diff2_motor = np.power(diff_motor,2)
    loss_motor = diff2_motor/np.power(mean_dict[key+"_motor_std"],2)
    loss_motor = np.max(np.mean(loss_motor,axis=0))

    diff_drpy = drpy-mean_dict[key+"_drpy_mean"]
    diff2_drpy = np.power(diff_drpy,2)
    loss_drpy = diff2_drpy/np.power(mean_dict[key+"_drpy_std"],2)
    loss_drpy = np.max(np.mean(loss_drpy,axis=0))

    return (loss_drpy+loss_motor)/2.0

@parl.remote_class(wait=False)
class RemoteESAgent(object):
    def __init__(self,mean_dict,gait,id):
        self.mean_dict = copy(mean_dict)
        self.gait = gait
        self.id = id
        self.env =  rlschool.make_env('Quadrupedal',task="ground",render=False,ETG=0)
        self.pose_ori = np.array([0,0.9,-1.8]*4)
        print("Init thread {}!".format(id))

    def sample_episode(self,param=None,e_step=300,key="exp"):
        dynamic_param = param2dynamic_dict(param)
        obs,info = self.env.reset(hardset=False,dynamic_param=dynamic_param)
        steps = 0
        motor_list = []
        drpy_list = []
        for i in range(100):
            action = self.gait[key][i]-self.pose_ori
            next_obs, reward, done, info = self.env.step(action,donef=False)
            steps += 1
            motor_list.append(info["joint_angle"])
            drpy_list.append(info["obs-IMU"][3:])
        motor_list = np.asarray(motor_list).reshape(-1,12)
        loss = loss_func(drpy_list,motor_list,self.mean_dict,key)
        reward = 30-loss
        return reward,self.id
        
    def batch_sample_episodes(self,param=None,K=1):
        returns = []
        for i in range(K):
            reward1,_ = self.sample_episode(param=param[i],key="exp")
            reward2,_ = self.sample_episode(param=param[i],key="ori")
            reward = (reward1+reward2)/2.0
            returns.append((reward,self.id*K+i))
        return returns

class ES_ParallelModel():
    def __init__(self,mean_dict,gait,num_params=48,K=4,thread=10,sigma=0.1,sigma_decay=0.9999,
                    dynamic_param=None,outdir="Results",
                    alg = "ga",xparl_addr="172.18.188.17:6006"):

        self.dynamic_param = dynamic_param
        self.sigma = sigma
        self.sigma_decay = sigma_decay
        self.num_params = num_params
        self.outdir = outdir
        self.alg = alg
        self.K = K
        self.thread = thread
        self.popsize = K*thread
        print("numparams:",self.num_params)
        print("addr:{}".format(xparl_addr))
        parl.connect(xparl_addr)
        self.agent_list = [
            RemoteESAgent(mean_dict=mean_dict,gait=gait,id = i) 
            for i in range(self.thread)
        ]
        self.set_solver(param=np.zeros(self.num_params))

    def set_solver(self,param=None):
        if self.alg == "ga":
            self.solver = SimpleGA(self.num_params,
                sigma_init=self.sigma,
                sigma_decay=self.sigma_decay,
                sigma_limit=0.02,
                elite_ratio=0.1,
                weight_decay=0.005,
                popsize=self.popsize,
                param = param)
        elif self.alg == "ses":
            self.solver = PEPG(self.num_params,
                sigma_init=self.sigma,
                sigma_decay=self.sigma_decay,
                sigma_alpha=0.2,
                sigma_limit=0.02,
                elite_ratio=0.1,
                weight_decay=0.005,
                popsize=self.popsize)
        elif self.alg == "pepg":
            self.solver = PEPG(self.num_params,
                sigma_init=self.sigma,
                sigma_decay=self.sigma_decay,
                sigma_alpha=0.20,
                sigma_limit=0.02,
                learning_rate=0.01,
                learning_rate_decay=1.0,
                learning_rate_limit=0.01,
                weight_decay=0.005,
                popsize=self.popsize)
        elif self.alg == "openes":
            self.solver = OpenES(self.num_params,
                sigma_init=self.sigma,
                sigma_decay=self.sigma_decay,
                sigma_limit=0.02,
                learning_rate=0.01,
                learning_rate_decay=1.0,
                learning_rate_limit=0.01,
                antithetic=True,
                weight_decay=0.005,
                popsize=self.popsize)
        elif self.alg == "simples":
            self.solver = SimpleES(self.num_params,
                sigma_init = self.sigma,
                sigma_decay = self.sigma_decay,
                sigma_limit = 0.02,
                weight_decay = 0.005,
                popsize = self.popsize)
    def save(self,epoch,outdir):
        np.save(os.path.join(outdir,"dynamic_param{}.npy".format(epoch)),self.dynamic_param)
    def update(self,epoch):
        rewards = []
        solutions = self.solver.ask()
        fitness_list = np.zeros(self.solver.popsize)
        future_objects = []
        for i in range(self.thread):
            future_objects.append(self.agent_list[i].batch_sample_episodes(param=solutions[i*self.K:(i+1)*self.K,:],
                                                                    K = self.K))
        results_list = [future_obj.get() for future_obj in future_objects]
        for i in range(self.thread):
            results = results_list[i]
            for j in range(self.K):
                result = results[j]
                rewards.append(result[0])
                fitness_list[i*self.K+j] = result[0]
        self.solver.tell(fitness_list)
        result = self.solver.result()
        sig = np.mean(result[3])
        max_index = np.argmax(rewards)
        self.dynamic_param = result[0]
        self.save(epoch,self.outdir)
        mean_reward = np.mean(rewards)
        #logger
        logger.info('Steps: {} Reward: {} sigma:{}'.format(epoch, np.max(rewards),sig))
        # logger.info('info:{}'.format(infos))
        summary.add_scalar('train/sigma', sig, epoch)
        summary.add_scalar('train/episode_reward', mean_reward, epoch)
        summary.add_scalar('train/episode_minre', np.min(rewards), epoch)
        summary.add_scalar('train/episode_maxre', np.max(rewards), epoch)
        summary.add_scalar('train/episode_restd', np.std(rewards), epoch)
        return result[1]
    def train(self,epochs):
        step_interval = 0.001
        step_min = 4
        for epoch in range(epochs):
            mean_re = self.update(epoch) 
            if epoch % 5 ==0 and epoch > 0:
                r = self.evaluate_episode(epoch)
                self.save(epoch,self.outdir)
    def evaluate_episode(self,epoch):
        results = self.agent_list[0].sample_episode(param=self.dynamic_param,key="height")
        reward,_ = results.get()
        logger.info('Steps: {} Eval Reward: {}'.format(epoch, reward))
        summary.add_scalar('eval/episode_reward', reward, epoch)
        return reward   




