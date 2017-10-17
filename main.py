from __future__ import print_function

import argparse
import os

import subprocess
import torch
import torch.multiprocessing as mp

import my_optim
from envs import create_atari_env
from model import ActorCritic
from test import test
from train import train

# Based on
# https://github.com/pytorch/examples/tree/master/mnist_hogwild
# Training settings
parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--tau', type=float, default=1.00,
                    help='parameter for GAE (default: 1.00)')
parser.add_argument('--entropy-coef', type=float, default=0.01,
                    help='entropy term coefficient (default: 0.01)')
parser.add_argument('--value-loss-coef', type=float, default=0.5,
                    help='value loss coefficient (default: 0.5)')
parser.add_argument('--max-grad-norm', type=float, default=50,
                    help='value loss coefficient (default: 50)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--num-processes', type=int, default=4,
                    help='how many training processes to use (default: 4)')
parser.add_argument('--num-steps', type=int, default=20,
                    help='number of forward steps in A3C (default: 20)')
parser.add_argument('--max-episode-length', type=int, default=1000000,
                    help='maximum length of an episode (default: 1000000)')
parser.add_argument('--env-name', default='PongDeterministic-v4',
                    help='environment to train on (default: PongDeterministic-v4)')
parser.add_argument('--no-shared', default=False,
                    help='use an optimizer without shared momentum.')


'''if delete logdir and start a new run'''
CLEAR_RUN = True 

'''display a tag before the result printed, to identify multiple runs on your machine'''
MULTI_RUN = 'GTN_1'

# create params and manage function
params = {}
params_seq = []
def add_parameters(**kwargs):
    global params_seq
    params_seq += kwargs.keys()
    params.update(kwargs)

# the first level of log dir
add_parameters(EXP = 'exp_1')

add_parameters(TIME_TO = 6*60*60)

'''
summary settings
to create log dir and params, do not modify if not
necessary.
'''
DSP = ''
params_str = 'Settings'+'\n'
params_str += '##################################'+'\n'
for i in range(len(params_seq)):
    DSP += params_seq[i]+'_'+str(params[params_seq[i]]).replace('.','_').replace(',','_').replace(' ','_')+'/'
    params_str += params_seq[i]+' >> '+str(params[params_seq[i]])+'\n'
params_str += '##################################'+'\n'
print(params_str)

'''
build log dir
'''
BASIC = '../../result/'
LOGDIR = BASIC+DSP
if CLEAR_RUN:
    subprocess.call(["rm", "-r", LOGDIR])
subprocess.call(["mkdir", "-p", LOGDIR])
with open(LOGDIR+"Settings.txt","a") as f:
    f.write(params_str)


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'

    args = parser.parse_args()

    # uncomment when it's fixed in pytorch
    # torch.manual_seed(args.seed)

    env = create_atari_env(args.env_name)
    shared_model = ActorCritic(
        env.observation_space.shape[0], env.action_space)
    shared_model.share_memory()

    if args.no_shared:
        optimizer = None
    else:
        optimizer = my_optim.SharedAdam(shared_model.parameters(), lr=args.lr)
        optimizer.share_memory()

    processes = []

    p = mp.Process(target=test, args=(args.num_processes, args, shared_model,LOGDIR,DSP,params_str,MULTI_RUN))
    p.start()
    processes.append(p)

    for rank in range(0, args.num_processes):
        p = mp.Process(target=train, args=(rank, args, shared_model, optimizer))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
