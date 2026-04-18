from sumolib import checkBinary
import gymnasium as gym
from env import TRAIN_CONFIG, ENV_CONFIG

import dqn.agent as Agent
import numpy as np

import os
import time
import argparse
import itertools
from datetime import timedelta
import torch
import traci


class Test:
    def __init__(self, args):
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        conf = 'config/highway.sumocfg'
        #args.gui = True,
        if args.gui:
            sumobin=checkBinary('sumo-gui')
            params = [sumobin, '-c', conf, "--delay", "500", "--collision.mingap-factor", "0", "--quit-on-end"]
        else:
            sumobin=checkBinary('sumo')
            params = [sumobin, '-c', conf, "--collision.mingap-factor", "0", "--no-step-log", "true",]  
        
        self.env = gym.make("PlatoonEnv-v0", params=params, gui=args.gui)

        self.agent = getattr(Agent, args.algo)(
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            gamma=args.gamma,

            bs=args.bs,
            clip_eps=args.clip_eps,
            gae_lambda=args.gae_lambda,
            epochs=args.epochs,

            input_dim=ENV_CONFIG['observation_space_size'],
            output_dim=ENV_CONFIG['action_space_size'],
            save_frequency=args.save_freq,
            log_frequency=args.log_freq,
            save_dir=args.save_dir,
            log_dir=args.log_dir,
            load=args.load,
            algo=args.algo,
            gpu=args.gpu
        )

        self.agent.load_model_test()
        self.max_total_episodes_test = args.max_total_episodes_test

        if torch.cuda.is_available():
            print("GPU is available")
        else: 
            print("GPU is not available")

        print()
        print("TEST")
        print()
        print(args.algo)
        print()
        [print(arg, "=", getattr(args, arg)) for arg in vars(args)]



    def test_loop(self):
        print()
        print("Start Testing")
        episode = 1
        observation, info = self.env.reset()
        for step in itertools.count(start=self.agent.resume_step):
            self.agent.step = step

            action = self.agent.choose_action_test(observation)
            
            new_observation, reward, terminated, truncated, info = self.env.step(action)

            done = terminated or truncated

            self.agent.store_transition_test(observation, action, reward, done, info)
            
            if done:                
                observation, _ = self.env.reset()
                episode += 1
            else: 
                observation = new_observation

            self.agent.log_test()

            if bool(self.max_total_episodes_test) and episode >= self.max_total_episodes_test:
                exit()

    def run(self):
        self.test_loop()
        traci.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TEST")
    str2bool = (lambda v: v.lower() in ("yes", "y", "true", "t", "1"))
    parser.add_argument('-gui', action='store_true', help='Enable GUI mode')
    parser.add_argument('-gpu', type=str, default=TRAIN_CONFIG["gpu"], help='GPU #')
    parser.add_argument('-n_env', type=int, default=TRAIN_CONFIG["n_env"], help='Multi-processing environments')
    parser.add_argument('-actor_lr', type=float, default=TRAIN_CONFIG["actor_lr"], help='Learning rate')
    parser.add_argument('-critic_lr', type=float, default=TRAIN_CONFIG["critic_lr"], help='Learning rate')
    parser.add_argument('-gamma', type=float, default=TRAIN_CONFIG["gamma"], help='Discount factor')
    parser.add_argument('-bs', type=int, default=TRAIN_CONFIG["bs"], help='Batch size')
    parser.add_argument('-clip_eps', type=int, default=TRAIN_CONFIG["clip_eps"], help='Batch size')
    parser.add_argument('-gae_lambda', type=int, default=TRAIN_CONFIG["gae_lambda"], help='Batch size')
    parser.add_argument('-epochs', type=int, default=TRAIN_CONFIG["epochs"], help='Batch size')
    parser.add_argument('-save_freq', type=int, default=TRAIN_CONFIG["save_freq"], help='Save frequency')
    parser.add_argument('-log_freq', type=int, default=TRAIN_CONFIG["log_freq"], help='Log frequency')
    parser.add_argument('-save_dir', type=str, default=TRAIN_CONFIG["save_dir"], help='Save directory')
    parser.add_argument('-log_dir', type=str, default=TRAIN_CONFIG["log_dir_test"], help='Log directory')
    parser.add_argument('-load', type=str2bool, default=TRAIN_CONFIG["load"], help='Load model')
    parser.add_argument('-repeat', type=int, default=TRAIN_CONFIG["repeat"], help='Steps repeat action')
    parser.add_argument('-max_episode_steps', type=int, default=TRAIN_CONFIG["max_episode_steps"], help='Episode step limit')
    parser.add_argument('-max_total_steps', type=int, default=TRAIN_CONFIG["max_total_steps"], help='Max total training steps')
    parser.add_argument('-max_total_episodes_test', type=int, default=TRAIN_CONFIG["max_total_episodes_test"],
                        help='Max total testing episodes')
    parser.add_argument('-algo', type=str, default=TRAIN_CONFIG["algo"],
                        help= 'PPORLAgent '
                        )

    Test(parser.parse_args()).run()
