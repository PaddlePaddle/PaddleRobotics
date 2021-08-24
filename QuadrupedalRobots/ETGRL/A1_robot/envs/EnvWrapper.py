import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
from robots import a1_robot
from robots import robot_config
from robots import action_filter
import collections
import numpy as np
import time
from utilities.Bezier import BezierGait
from utilities.SpotOL import BezierStepper
from copy import copy
mode_map ={"pose":robot_config.MotorControlMode.POSITION,
            "torque":robot_config.MotorControlMode.TORQUE,
            "traj":robot_config.MotorControlMode.POSITION}
from tqdm import tqdm
def EnvWrapper(robot,dt,sensor_mode,gait,normal,enable_action_filter,CPG_data):
    env = SimpleEnv(robot,sensor_mode,normal,CPG_data)
    if gait:
        env = GaitWrapper(env,robot,gait,dt)
    env = ObservationWrapper(env,robot,sensor_mode)
    env = ActionFilterWrapper(env,robot,dt,enable_action_filter)
    # env = ControlLoopWrapper(env,dt)
    return env


class SimpleEnv(object):
    def __init__(self,robot,sensor_mode,normal,CPG_data=None):
        self.robot = robot
        self.sensor_mode = sensor_mode
        self.obs_dim = 0
        self.last_action = np.array([0,0.9,-1.8]*4)
        self.normal = normal
        self.CPG_data = CPG_data
        if "motor" in self.sensor_mode:
            if self.sensor_mode["motor"] == 1:
                self.obs_dim += 24
            elif self.sensor_mode["motor"] == 2:
                self.obs_dim += 12
        if "dis" in self.sensor_mode and self.sensor_mode["dis"]:
            self.obs_dim += 3
        if "imu" in self.sensor_mode and self.sensor_mode["imu"]:
            self.obs_dim += 6
        if "contact" in self.sensor_mode and self.sensor_mode["contact"]:
            self.obs_dim += 4
        if "CPG" in self.sensor_mode and self.sensor_mode["contact"]:
            self.obs_dim += 12

        self.CPG_mean = np.array([2.1505982e-02,  3.6674485e-02, -6.0444288e-02,
                        2.4625482e-02,  1.5869144e-02, -3.2513142e-02,  2.1506395e-02,
                        3.1869926e-02, -6.0140789e-02,  2.4625063e-02,  1.1628972e-02,
                        -3.2163858e-02])
        self.CPG_std = np.array([4.5967497e-02,2.0340437e-01, 3.7410179e-01, 4.6187632e-02, 1.9441207e-01, 3.9488649e-01,
                                4.5966785e-02 ,2.0323379e-01, 3.7382501e-01, 4.6188373e-02 ,1.9457331e-01, 3.9302582e-01])

    def get_obs_dim(self):
        return self.obs_dim

    def get_observation(self):
        self.robot.ReceiveObservation()
        sensors_dict = {}
        if "motor" in self.sensor_mode:
            motor_angle = self.robot.GetMotorAngles()
            if self.normal:
                motor_angle = (np.array(motor_angle)-np.array([0,0.9,-1.8]*4))/0.1
            if self.sensor_mode["motor"] == 1:
                motor_vel = self.robot.GetMotorVelocities()
                if self.normal:
                    motor_vel = (np.array(motor_vel)-np.zeros(12))/1.0
                sensors_dict["MotorAngleAcc"] = np.concatenate((motor_angle,motor_vel),axis=0)
            elif self.sensor_mode["motor"] == 2:
                sensors_dict["MotorAngle"] = motor_angle

        if "dis" in self.sensor_mode and self.sensor_mode["dis"]:
            sensors_dict["BaseDisplacement"] = self.robot.GetBaseVelocity()

        if "imu" in self.sensor_mode:
            if self.first_reset:
                self.first_rpy = np.array(self.robot.GetTrueBaseRollPitchYaw())
                rpy = np.zeros(3)
                self.first_reset = False
            else:
                rpy = np.array(self.robot.GetTrueBaseRollPitchYaw())-self.first_rpy
            drpy = self.robot.GetTrueBaseRollPitchYawRate()
            if self.normal:
                rpy = (rpy-np.zeros(3))/0.1
                drpy = (np.array(drpy)-np.zeros(3))/0.5
            if self.sensor_mode["imu"]==1:
                sensors_dict["IMU"] = np.concatenate((rpy,drpy),axis=0)
            elif self.sensor_mode["imu"]==2:
                sensors_dict["IMU"] = drpy

        if "contact" in self.sensor_mode and self.sensor_mode["contact"]:
            contact = self.robot.GetFootContacts()
            sensors_dict["FootContactSensor"] = contact

        observation_dict = collections.OrderedDict(sorted(list(sensors_dict.items())))
        observations = []
        for key, value in observation_dict.items():
            observations.append(np.asarray(value).flatten())
        flat_observations = np.concatenate(observations)
        if "CPG" in self.sensor_mode and self. sensor_mode["CPG"]:
            CPG_output = self.CPG_data[self.iter]
            if self.normal:
                CPG_output = (CPG_output-self.CPG_mean)/self.CPG_std
            flat_observations = np.concatenate((flat_observations,CPG_output),axis=0)
        observation_dict["real_action"] = self.last_action
        return flat_observations,observation_dict

    def reset(self,**kwargs):
        self.first_reset = True
        self.iter = 0
        self.robot.Reset()     
        obs,info = self.get_observation()
        return obs,info
    
    def step(self,action,**kwargs):
        self.robot.Step(action, robot_config.MotorControlMode.POSITION)
        self.last_action = action
        self.iter += 1
        # obs,info = self.get_observation()
        # info["real_action"] = action
        # return obs,info

class GaitWrapper(object):
    def __init__(self,env,robot,gait,dt,velocity=0.5):
        self.env = env
        self.gait = gait
        self.robot = robot
        self.timesteps = 0
        self.dt = dt
        self.velocity = velocity
        self.info = {}
        self.obs_dim = self.env.get_obs_dim()

    def get_obs_dim(self):
        return self.obs_dim

    def get_observation(self):
        return self.env.get_observation()

    def reset(self,**kwargs):
        obs,info = self.env.reset(**kwargs)
        self.timesteps =0 
        self.bz_step = BezierStepper(dt=self.dt,StepVelocity=self.velocity)
        self.bzg = BezierGait(dt=self.dt)
        T_b0_ = copy(self.robot.GetFootPositionsInBaseFrame())
        Tb_d = {}
        Tb_d["FL"]=T_b0_[0,:]
        Tb_d["FR"]=T_b0_[1,:]
        Tb_d["BL"]=T_b0_[2,:]
        Tb_d["BR"]=T_b0_[3,:]
        self.T_b0 = Tb_d
        self.info = info
        return obs,info
    
    def step(self,action,**kwargs):
        t_start = time.clock()
        self.timesteps += 1
        pos, orn, StepLength, LateralFraction, YawRate, StepVelocity, ClearanceHeight, PenetrationDepth = self.bz_step.StateMachine()
        ClearanceHeight = 0.05
        StepLength = np.clip(StepLength, self.bz_step.StepLength_LIMITS[0],
                                    self.bz_step.StepLength_LIMITS[1])
        StepVelocity = np.clip(StepVelocity,
                                    self.bz_step.StepVelocity_LIMITS[0],
                                    self.bz_step.StepVelocity_LIMITS[1])
        LateralFraction = np.clip(LateralFraction,
                                    self.bz_step.LateralFraction_LIMITS[0],
                                    self.bz_step.LateralFraction_LIMITS[1])
        YawRate = np.clip(YawRate, self.bz_step.YawRate_LIMITS[0],
                                    self.bz_step.YawRate_LIMITS[1])
        ClearanceHeight = np.clip(ClearanceHeight,
                                    self.bz_step.ClearanceHeight_LIMITS[0],
                                    self.bz_step.ClearanceHeight_LIMITS[1])
        PenetrationDepth = np.clip(PenetrationDepth,
                                    self.bz_step.PenetrationDepth_LIMITS[0],
                                    self.bz_step.PenetrationDepth_LIMITS[1])
        contacts = copy(self.info["FootContactSensor"])
        if self.timesteps > 5:
            T_bf = self.bzg.GenerateTrajectoryX(StepLength, LateralFraction,
                                                YawRate, StepVelocity, self.T_b0, ClearanceHeight,
                                                PenetrationDepth, contacts)
        else:
            T_bf = self.bzg.GenerateTrajectoryX(0.0, 0.0, 0.0, 1, self.T_b0, ClearanceHeight,
                                                PenetrationDepth, contacts)
        # print("gait_t_now:",time.clock()-t_start)
        leg_id = 0
        action_ref = np.zeros(12)
        for key in T_bf:
            leg_pos = T_bf[key]
            index, angle = self.robot.ComputeMotorAnglesFromFootLocalPosition(leg_id,leg_pos)
            action_ref[index] = np.asarray(angle)
            leg_id += 1
        new_action = action + action_ref
        # print("gait_t_all:",time.clock()-t_start)
        self.env.step(new_action,**kwargs)
        self.info = info

class ObservationWrapper(object):
    def __init__(self,env,robot,sensor_mode):
        self.env = env
        self.robot = robot 
        self.sensor_mode = sensor_mode
        self.obs_dim = self.env.get_obs_dim()
        if "RNN" in self.sensor_mode.keys() and self.sensor_mode["RNN"]["time_steps"]>0 and self.sensor_mode["RNN"]["mode"]=="stack":
            self.obs_dim *= (self.sensor_mode["RNN"]["time_steps"]+1)

    def get_obs_dim(self):
        return self.obs_dim

    def get_observation(self):
        obs,info = self.env.get_observation()
        if "RNN" in self.sensor_mode.keys() and self.sensor_mode["RNN"]["time_steps"]>0:
            obs_list = []
            for t in range(self.time_steps):
                obs_list.append(copy(self.obs_history[t*self.time_interval]))
            obs_list.append(copy(obs))
            self.obs_history[:-1] = copy(self.obs_history[1:])
            self.obs_history[-1] = copy(obs)
            if self.sensor_mode["RNN"]["mode"]=="GRU":
                obs = np.stack(obs_list,axis=0)
            elif self.sensor_mode["RNN"]["mode"]=="stack":
                obs = np.array(obs_list).reshape(-1)
        return obs,info

    def reset(self,**kwargs):
        obs,info = self.env.reset(**kwargs)
        if "RNN" in self.sensor_mode.keys() and self.sensor_mode["RNN"]["time_steps"]>0:
            self.time_steps = self.sensor_mode["RNN"]["time_steps"]
            self.time_interval = self.sensor_mode["RNN"]["time_interval"]
            self.sensor_shape = obs.shape[0]
            self.obs_history = np.zeros((self.time_steps*self.time_interval,self.sensor_shape))
            obs_list = []
            for t in range(self.time_steps):
                obs_list.append(copy(self.obs_history[t*self.time_interval]))
            obs_list.append(copy(obs))
            self.obs_history[-1] = copy(obs)
            if self.sensor_mode["RNN"]["mode"]=="GRU":
                obs = np.stack(obs_list,axis=0)
            elif self.sensor_mode["RNN"]["mode"]=="stack":
                obs = np.array(obs_list).reshape(-1)
        return obs,info
    
    def step(self,action,**kwargs):
        self.env.step(action,**kwargs)

class ControlLoopWrapper(object):
    def __init__(self,env,dt,action_repeat):
        self.env = env
        self.dt = dt
        self.t0 = 0
        self.obs_dim = self.env.get_obs_dim()

    def get_obs_dim(self):
        return self.obs_dim

    def reset(self,**kwargs):
        obs,info = self.env.reset(**kwargs)
        self.t0 = time.clock()
        return obs,info
    
    def get_time(self):
        return time.clock()-self.t0
    
    def step(self,action,**kwargs):
        t_start = time.clock()
        obs,info = self.env.step(action,**kwargs)
        t_now = time.clock()-t_start
        print("t_now:",t_now)
        if t_now<self.dt-5e-4:
            time.sleep(self.dt-t_now)
        return obs,info


class ActionFilterWrapper(object):
    def __init__(self,env,robot,dt,enable_action_filter):
        self.env = env
        self.dt = dt
        self.robot = robot
        self.enable_action_filter = enable_action_filter
        if self.enable_action_filter:
            self._action_filter = self._BuildActionFilter()

    def reset(self,**kwargs):
        obs_all,info = self.env.reset(**kwargs)
        self._step_counter = 0
        if self.enable_action_filter:
            self._ResetActionFilter()
        return obs_all,info
    
    def step(self,action,**kwargs):
        if self.enable_action_filter:
            action = self._FilterAction(action)
        self.env.step(action,**kwargs)
        self._step_counter += 1
        # return obs_all, info
    
    def get_observation(self):
        return self.env.get_observation()


    def _BuildActionFilter(self):
        sampling_rate = 1 / self.dt
        num_joints = 12
        a_filter = action_filter.ActionFilterButter(sampling_rate=sampling_rate,
                                                    num_joints=num_joints)
        return a_filter

    def _ResetActionFilter(self):
        self._action_filter.reset()

    def _FilterAction(self, action):
        # initialize the filter history, since resetting the filter will fill
        # the history with zeros and this can cause sudden movements at the start
        # of each episode
        if self._step_counter == 0:
            default_action = np.array([0,0.9,-1.8]*4)
            self._action_filter.init_history(default_action)
            # for j in range(10):
            #     self._action_filter.filter(default_action)

        filtered_action = self._action_filter.filter(action)
        # print(filtered_action)
        return filtered_action    



      
        





