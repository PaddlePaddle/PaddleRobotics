from absl import flags
from absl import app
from absl import logging
import numpy as np
import time
from tqdm import tqdm
import pybullet  # pytype:disable=import-error
import pybullet_data
from pybullet_utils import bullet_client

from robots import a1_robot
from robots import robot_config
from envs import EnvWrapper
from model.mujoco_model import MujocoModel
from model.mujoco_agent import MujocoAgent
from model.sac import SAC
from copy import copy
import argparse

GAMMA = 0.99
TAU = 0.005
ALPHA = 0.2  # determines the relative importance of entropy term against the reward
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4

def get_obs_dim(sensor_mode):
    obs_dim = 0
    if "motor" in sensor_mode:
        if sensor_mode["motor"] == 1:
            obs_dim += 24
        elif sensor_mode["motor"] == 2:
            obs_dim += 12
    if "dis" in sensor_mode and sensor_mode["dis"]:
        obs_dim += 3
    if "imu" in sensor_mode:
        if sensor_mode["imu"] == 1:
            obs_dim += 6
        elif sensor_mode["imu"] == 2:
            obs_dim += 3
    if "contact" in sensor_mode and sensor_mode["contact"]:
        obs_dim += 4
    if "ETG" in sensor_mode and sensor_mode["ETG"]:
        obs_dim += 12
    if "RNN" in sensor_mode.keys() and sensor_mode["RNN"]["time_steps"]>0 and sensor_mode["RNN"]["mode"]=="stack":
        obs_dim *= (sensor_mode["RNN"]["time_steps"]+1)
    return obs_dim

def main():
    logging.info(
        "WARNING: this code executes low-level controller on the robot.")
    logging.info("Make sure the robot is hang on rack before proceeding.")
    input("Press enter to continue...")

    sensor_mode = {}
    sensor_mode['dis'] = args.sensor_dis
    sensor_mode['motor'] = args.sensor_motor
    sensor_mode["imu"] = args.sensor_imu
    sensor_mode["contact"] = args.sensor_contact
    sensor_mode["ETG"] = args.sensor_ETG
    rnn_config = {}
    rnn_config["time_steps"] = args.timesteps
    rnn_config["time_interval"] = args.timeinterval
    rnn_config["mode"] = args.RNN_mode
    sensor_mode["RNN"] = rnn_config
    obs_dim = get_obs_dim(sensor_mode)
    action_dim = 12
    act_bound = np.array([0.3,0.3,0.3]*4)
    model = MujocoModel(obs_dim, action_dim)
    algorithm = SAC(
        model,
        gamma=GAMMA,
        tau=TAU,
        alpha=ALPHA,
        actor_lr=ACTOR_LR,
        critic_lr=CRITIC_LR,
        joint_rate = 0.0)
    agent = MujocoAgent(algorithm)
    agent.restore(args.load)
    obs_list = []
    action_list = []
    action = agent.predict(np.zeros(obs_dim))*act_bound
    # Construct sim env and real robot
    p = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    gait_action = np.load(args.ETG_path)
    robot = a1_robot.A1Robot(pybullet_client=p, action_repeat=1)
    env = EnvWrapper.EnvWrapper(robot=robot,dt=args.dt,sensor_mode=sensor_mode,gait=args.gait,
                                normal= args.normal,enable_action_filter=args.enable_action_filter,
                                ETG_data = copy(gait_action))
    obs,info = env.reset()
    base_act = np.array([0,0.9,-1.8]*4)
    for i in range(int(args.max_time*100)):
        t_start = time.clock()
        ref_action = gait_action[i]
        action = agent.predict(obs)*act_bound+ref_action
        obs_list.append(obs)
        action_list.append(action)
        env.step(base_act+action)
        obs,info = env.get_observation()
        t_now = time.clock()-t_start
        if args.dt-t_now >= 5e-4:
            time.sleep(args.dt-t_now)
    
    np.savez("data/"+args.suffix+"_rpm.npz",action=action_list,obs=obs_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix",type=str,default="exp0")
    parser.add_argument("--ETG_path",type=str,default="exp/stair_6_21/gait_action_list_ETG_stair.npy")
    parser.add_argument("--sensor_dis",type=int,default=0)
    parser.add_argument("--sensor_motor",type=int,default=1)
    parser.add_argument("--sensor_imu",type=int,default=1)
    parser.add_argument("--sensor_contact",type=int,default=1)
    parser.add_argument("--sensor_footpose",type=int,default=0)
    parser.add_argument("--sensor_ETG",type=int,default=1)
    parser.add_argument("--timesteps",type=int,default=5)
    parser.add_argument("--timeinterval",type=int,default=1)
    parser.add_argument("--RNN_mode",type=str,default="None")
    parser.add_argument("--dt",type=float,default=0.026)
    parser.add_argument("--max_time",type=float,default=1)
    parser.add_argument("--normal",type=float,default=1)
    parser.add_argument("--gait",type=int,default=0)
    parser.add_argument("--load", type=str, default="", help="Directory to load agent from.")
    parser.add_argument("--enable_action_filter",type=int,default=0)
    args = parser.parse_args()
    main()



