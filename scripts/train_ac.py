from tqdm import tqdm
from time import time
import torch
from torch.utils.data import DataLoader
import argparse
import numpy as np

from model.agents import *
from env import *

import utils


if __name__ == '__main__':
    
    # initial args
    init_parser = argparse.ArgumentParser()
    init_parser.add_argument('--env_class', type=str, default='RL4RSEnvironment', help='Environment class.')
    init_parser.add_argument('--policy_class', type=str, default='RL4RS_OneStage_Policy', help='Policy class')
    init_parser.add_argument('--critic_class', type=str, default='GeneralCritic', help='Critic class')
    init_parser.add_argument('--agent_class', type=str, default='DDPG', help='Learning agent class')
    init_parser.add_argument('--facade_class', type=str, default='RL4RSFacade', help='Environment class.')
    
    initial_args, _ = init_parser.parse_known_args()
    print(initial_args)
    
    envClass = eval('{0}.{0}'.format(initial_args.env_class))
    policyClass = eval('policy.{0}'.format(initial_args.policy_class))
    criticClass = eval('critic.{0}'.format(initial_args.critic_class))
    agentClass = eval('{0}.{0}'.format(initial_args.agent_class))
    facadeClass = eval('{0}.{0}'.format(initial_args.facade_class))
    
    # control args
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=11, help='random seed')
    
    # customized args
    parser = envClass.parse_model_args(parser)
    parser = policyClass.parse_model_args(parser)
    parser = criticClass.parse_model_args(parser)
    parser = agentClass.parse_model_args(parser)
    parser = facadeClass.parse_model_args(parser)
    args, _ = parser.parse_known_args()
    
    utils.set_random_seed(args.seed)
    
    # Environment
    print("Loading environment")
    env = envClass(args)
    
    # Agent
    device = 'cpu'
    print("Setup policy:")
    policy = policyClass(args, env)
    print(policy)
    print("Setup critic:")
    critic = criticClass(args, env, policy)
    print(critic)
    print("Setup agent with data-specific facade")
    facade = facadeClass(args, env, policy, critic)
    agent = agentClass(args, facade)
    
    try:
        print(args)
        agent.train()
    except KeyboardInterrupt:
        print("Early stop manually")
        exit_here = input("Exit completely without evaluation? (y/n) (default n):")
        if exit_here.lower().startswith('y'):
            print(os.linesep + '-' * 20 + ' END: ' + utils.get_local_time() + ' ' + '-' * 20)
            exit(1)
    
    
    