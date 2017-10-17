import time
from collections import deque

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from envs import create_atari_env
from model import ActorCritic

import plot
import numpy as np


def test(rank, args, shared_model, LOGDIR,DSP,params_str,MULTI_RUN):

    '''build and try restore logger'''
    logger = plot.logger(LOGDIR,DSP,params_str,MULTI_RUN)
    iteration = logger.restore()

    torch.manual_seed(args.seed + rank)

    env = create_atari_env(args.env_name)
    env.seed(args.seed + rank)

    model = ActorCritic(env.observation_space.shape[0], env.action_space)

    model.eval()

    state = env.reset()
    state = torch.from_numpy(state)
    reward_sum = 0
    done = True

    log_count = 0

    start_time = time.time()

    # a quick hack to prevent the agent from stucking
    actions = deque(maxlen=100)
    episode_length = 0
    while True:
        episode_length += 1
        # Sync with the shared model
        if done:
            model.load_state_dict(shared_model.state_dict())
            cx = Variable(torch.zeros(1, 256), volatile=True)
            hx = Variable(torch.zeros(1, 256), volatile=True)
        else:
            cx = Variable(cx.data, volatile=True)
            hx = Variable(hx.data, volatile=True)

        value, logit, (hx, cx) = model((Variable(
            state.unsqueeze(0), volatile=True), (hx, cx)))
        prob = F.softmax(logit)
        action = prob.max(1, keepdim=True)[1].data.numpy()

        state, reward, done, _ = env.step(action[0, 0])
        done = done or episode_length >= args.max_episode_length
        reward_sum += reward

        # a quick hack to prevent the agent from stucking
        actions.append(action[0, 0])
        if actions.count(actions[0]) == actions.maxlen:
            done = True

        if done:
            print("Time {}, episode reward {}, episode length {}".format(
                time.strftime("%Hh %Mm %Ss",
                              time.gmtime(time.time() - start_time)),
                reward_sum, episode_length))

            # print(reward_sum)

            logger.plot(
                'episode reward',
                np.asarray([reward_sum])
            )
            logger.plot(
                'episode length',
                np.asarray([float(episode_length)])
            )
            log_count += 1

            reward_sum = 0
            episode_length = 0
            actions.clear()
            state = env.reset()

            logger.tick()
            time.sleep(60)

            if log_count>3:
                logger.flush()

        state = torch.from_numpy(state)
