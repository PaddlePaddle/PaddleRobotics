# Third party code
#
# The following code are copied or modified from:
# https://github.com/google-research/motion_imitation
"""Apply the same action to the simulated and real A1 robot.


As a basic debug tool, this script allows you to execute the same action
(which you choose from the pybullet GUI) on the simulation and real robot
simultaneouly. Make sure to put the real robbot on rack before testing.
"""

from absl import app
from absl import flags
from absl import logging
import numpy as np
import time
from tqdm import tqdm
import pybullet  # pytype:disable=import-error
import pybullet_data
from pybullet_utils import bullet_client

from robots import a1_robot
from robots import robot_config
FREQ = 0.5

flags.DEFINE_string("suffix", "", "where to save file.")
FLAGS = flags.FLAGS

def main(_):
  logging.info(
      "WARNING: this code executes low-level controller on the robot.")
  logging.info("Make sure the robot is hang on rack before proceeding.")
  input("Press enter to continue...")

  # Construct sim env and real robot
  p = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)
  p.setAdditionalSearchPath(pybullet_data.getDataPath())
  robot = a1_robot.A1Robot(pybullet_client=p, action_repeat=1)

  # Move the motors slowly to initial position
  robot.ReceiveObservation()
  current_motor_angle = np.array(robot.GetMotorAngles())
  desired_motor_angle = np.array([0., 0.9, -1.8] * 4)
  for t in tqdm(range(300)):
    blend_ratio = np.minimum(t / 200., 1)
    action = (1 - blend_ratio
              ) * current_motor_angle + blend_ratio * desired_motor_angle
    robot.Step(action, robot_config.MotorControlMode.POSITION)
    time.sleep(0.005)

  vs = []
  motor_angles = []
  motor_velocities = []
  foot_contacts = []
  imus = []
  rpys = []
  actions = []
  # Move the legs in a sinusoidal curve
  for t in tqdm(range(1000)):
    t_start = time.clock()
    angle_hip = 0.9 + 0.2 * np.sin(2 * np.pi * FREQ * 0.01 * t)
    angle_calf = -2 * angle_hip
    action = np.array([0., 0.9, -1.8] * 4)
    robot.Step(action, robot_config.MotorControlMode.POSITION)
    robot.ReceiveObservation()
    t_now = time.clock()-t_start
    if 0.03-t_now>0:
      time.sleep(0.03-t_now)
    motor_angles.append(robot.GetTrueMotorAngles())
    motor_velocities.append(robot.GetMotorVelocities())
    foot_contacts.append(robot.GetFootContacts())
    vs.append(robot.GetBaseVelocity())
    imus.append(robot.GetBaseRollPitchYawRate())
    rpys.append(robot.GetTrueBaseRollPitchYaw())
    actions.append(action)
  np.savez(FLAGS.suffix+"_obs_sin.npz",motor_angle=motor_angles,
                    motor_velocity=motor_velocities,
                    foot_contact=foot_contacts,
                    v = vs,
                    imu = imus,
                    rpy = rpys,
                    action = actions)
    # time.sleep(0.007)
    # print(robot.GetFootContacts())
    # print(robot.GetBaseVelocity())

  robot.Terminate()

if __name__ == '__main__':
  app.run(main)
