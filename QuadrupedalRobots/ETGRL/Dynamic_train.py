import numpy as np
import gym
import sys
import os
import argparse
from model.Dynamic_parallel_model import ES_ParallelModel
from parl.utils import logger, summary
from copy import copy

MEAN_INFO = np.load("data/dynamic/mean_dict_5_18.npz")
MEAN_DICT = {}
for key in MEAN_INFO.keys():
    MEAN_DICT[key] = MEAN_INFO[key]
GAIT_LIST = {}
GAIT_LIST["exp"] = np.load("data/dynamic/gait_action_list_t0.3.npy")
GAIT_LIST["ori"] = np.load("data/dynamic/gait_action_list_CPG_ori.npy")
GAIT_LIST["height"] = np.load("data/dynamic/gait_action_list_CPG_height.npy")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir",type=str,default="Dynamic")
    parser.add_argument("--steps",type=int,default=10000)
    parser.add_argument("--K", type=int, default=20, help="Number of experiments run in one time.")
    parser.add_argument("--load", type=str, default="", help="Directory to load agent from.") #CPG_model_1/slip2/model_itr590.npz
    parser.add_argument("--eval", type=int, default=0, help="Evaluate or not")
    parser.add_argument("--suffix",type=str,default="exp0")
    parser.add_argument("--sigma",type=float,default=0.1)
    parser.add_argument("--gamma",type=float,default=1)
    parser.add_argument("--alg",type=str,default="ga",help="ga ses cma pepg openes")
    parser.add_argument("--xparl",type=str,default="172.18.188.13:8007")
    parser.add_argument("--thread",type=int,default=2)
    args = parser.parse_args()
    if len(args.load)>1:
        dynamic_param =np.load(args.load)
    else:
        dynamic_param = None
    outdir = os.path.join(args.outdir,args.suffix)
    if args.eval:
        args.thread = 1
    model = ES_ParallelModel(mean_dict=MEAN_DICT,gait=GAIT_LIST,
                K=args.K,thread = args.thread,dynamic_param=dynamic_param,outdir=outdir,
                alg=args.alg,sigma=args.sigma,xparl_addr = args.xparl)
    if args.eval:
        model.evaluate_episode(0)
    else:
        if not os.path.exists(args.outdir):
            os.makedirs(args.outdir)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        logger.set_dir(outdir)
        logger.info('args:{}'.format(args))
        model.train(args.steps)
if __name__ == '__main__':
    main()
    
