#!/usr/bin/env python3
import argparse
import os
import sys
from math import cos, exp, sin
import matplotlib.pyplot as plt
import numpy as np
current_file_path = os.path.dirname(__file__)

gym_setting_path = os.path.join(current_file_path, '../../gym_setting/mdp')
sys.path.append(os.path.abspath(gym_setting_path))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from mdp import (
    Actions,
    MarkovDecisionProcess,
    Policy,
    Rewards,
    States,
    StateTransitionProbability,
    ValueIteration,
    PolicyIteration
)
from scipy import sparse as sp
from scipy.stats.qmc import MultivariateNormalQMC
current_file_path = os.path.dirname(os.path.abspath(__file__))
import gym
from simulation.src.envs.dkc_unicycle_realUAV import DKC_real_Unicycle

class DKC_DP():
    def __init__(self, args):
        self.env = gym.make('DKC_real_Unicycle') if args.real else gym.make('DKC_Unicycle')
        self.n_sample = 2**8
        self.sample_reward = False
        self.n_r = round(self.env.r_max/self.env.v*10)
        print('n_r: ', self.n_r)
        self.n_alpha = 360
        self.n_u = 2 #args.num_actions #21

        # Parameters? Why??
        self.r_space = np.linspace(0, self.env.r_max, self.n_r, dtype=np.float64)  #<- replaced 0 with 1 considering computational domain
        self.alpha_space = np.linspace(-np.pi, np.pi - np.pi / self.n_alpha, self.n_alpha, dtype=np.float64)
        self.action_space = np.linspace(-1.0 / 4.5, 1.0 / 4.5, self.n_u, dtype=np.float64).reshape((-1, 1))
        self.sampler = MultivariateNormalQMC(mean=[0, 0], cov=[[1, 0], [0, 1]])
        self.sigma = self.env.v*self.env.dt/(args.sigma*np.sqrt(2)) # sigma*sqrt(2)=v*dt when sigma is covariance of gaussian noise
        # 1 sigma = vdt or 2sigma= vdt?

    def set_mdp(self, args):
        if args.load:
            load_dir = 'RESULTS/'+ args.load_dir
            mdp = MarkovDecisionProcess()
            self.mdp = mdp.load(load_dir)
        else:
            self.states = States(
                self.r_space, # distance
                self.alpha_space, # Heading Angle
                cycles=[np.inf, np.pi * 2], # Cycles
                n_alpha=self.n_alpha,
            )

            self.actions = Actions(
                self.action_space
            )

            state_transition_prob = StateTransitionProbability(
                states=self.states,
                actions=self.actions,
            )

            rewards = Rewards(self.states, self.actions)
            k1 = 0.0181  # 0.07273
            v = self.env.v
            print('v: ', v)
            d = self.env.d
            print('d: ', d)
            for idx, s in enumerate(self.states):
                rewards[idx, :] = -k1 * (s[0] - d) ** 2 - (-v * cos(s[1])) ** 2
                # rewards[idx, :] = (s[0] - d) ** 2

            policy = Policy(states=self.states, actions=self.actions)

            self.mdp = MarkovDecisionProcess(
                states=self.states,
                actions=self.actions,
                rewards=rewards,
                state_transition_probability=state_transition_prob,
                policy=policy,
                discount=0.99,
            )

    def noise_simulate(self, state):
        pose = np.array(
            [
                state[0] * cos(state[1] + np.pi),
                state[0] * sin(state[1] + np.pi),
                0.0,
            ],
            dtype=np.float64,
        )
        spmat = sp.dok_matrix(
            (self.actions.num_actions, self.states.num_states), dtype=np.float64
        )
        if self.sample_reward:
            arr = np.zeros((self.actions.num_actions), dtype=np.float64)
        for sample in self.sampler.random(self.n_sample):
            for action_idx, action in enumerate(self.actions):
                noisy_pose = pose.copy()
                noisy_pose[:2] += self.sigma * sample
                state = self.env.reset(pose=noisy_pose)
                if self.sample_reward:
                    next_state, reward, _, _, _ = self.env.step(action)
                    arr[action_idx] += reward / self.n_sample
                else:
                    next_state, _, _, _, _ = self.env.step(action)
                next_state_indices, probs = self.states.computeBarycentric(next_state)
                probs = probs.astype(np.float64)
                for next_state_idx, prob in zip(next_state_indices, probs):
                    spmat[action_idx, next_state_idx] += prob / self.n_sample
        if self.sample_reward:
            return spmat, arr
        else:
            return spmat

    def simulate(self, state):
        pose = np.array(
            [
                state[0] * cos(state[1] + np.pi),
                state[0] * sin(state[1] + np.pi),
                0.0,
            ],
            dtype=np.float64,
        )
        spmat = sp.dok_matrix(
            (self.actions.num_actions, self.states.num_states), dtype=np.float64
        )
        if self.sample_reward:
            arr = np.zeros((self.actions.num_actions), dtype=np.float64)
        for action_idx, action in enumerate(self.actions):
            state = self.env.reset(pose=pose.copy())
            if self.sample_reward:
                next_state, reward, _, _, _ = self.env.step(action)
                arr[action_idx] += reward / self.n_sample
            else:
                next_state, _, _, _, _ = self.env.step(action)
            next_state_indices, probs = self.states.computeBarycentric(next_state)
            probs = probs.astype(np.float64)
            for next_state_idx, prob in zip(next_state_indices, probs):
                spmat[action_idx, next_state_idx] += prob
        if self.sample_reward:
            return spmat, arr
        else:
            return spmat

    def run(self, args):
        if args.sigma:
            print('sample with noise')
            self.mdp.sample(self.noise_simulate, sample_reward=self.sample_reward, parallel=args.parallel)
        else:
            self.mdp.sample(self.simulate, sample_reward=self.sample_reward, parallel=args.parallel)
        print('sample successfull!')
        mdp_filename = f"RESULTS/dkc_r{self.env.r_max}_2a_sig{args.sigma}_mdp"
        self.mdp.save(mdp_filename)
        if args.solver == 'val':
            solver = ValueIteration(self.mdp)
        else:
            solver = PolicyIteration(self.mdp)
        solver.solve(max_iteration=10000, tolerance=1e-6, earlystop=100, save_name=f"RESULTS/dkc_r{self.env.r_max}_2a_sig{args.sigma}", parallel=args.parallel)

        # save moved to dynamic_programming.py>solver.solve
        value_filename = f"RESULTS/dkc_r{self.env.r_max}_2a_sig{args.sigma}_value.png"
        value_plot = solver.values.reshape((self.n_r, self.n_alpha))
        value_tips = np.flipud(value_plot.T)  # value plot of Tips paper format
        plt.imsave(value_filename, value_tips, cmap="gray")

        policy_filename = f"RESULTS/dkc_r{self.env.r_max}_2a_sig{args.sigma}_policy.png"
        policy_plot = solver.mdp.policy.toarray().reshape((self.n_r, self.n_alpha))
        policy_tips = np.flipud(policy_plot.T)  # policy plot of Tips paper format
        plt.imsave(
            policy_filename,
            policy_tips,
            cmap="gray",
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    def str2bool(v):
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false","f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser.add_argument(
        "-sol",
        "--solver",
        type=str,
        required=False,
        default="val",
        help="'val' for Value Iteration, 'pol' for Policy Iteration",
    )

    parser.add_argument(
        "-p",
        "--parallel",
        type=str2bool,
        required=False,
        default=True,
        help="parallel execution true or false",
    )
    
    parser.add_argument(
            "-nu",
            "--num_actions",
            type=int,
            required=False,
            default=2,
            help="number of actions, default: 21",
        )

    parser.add_argument(
            "-sig",
            "--sigma",
            type=float,
            required=False,
            default=0,
            help="noise covariance, default: 0",
        )
    parser.add_argument(
        "--load",
        type=str2bool,
        required=False,
        default=False,
        help="load mdp?",
    )
    parser.add_argument(
        "--load_dir",
        type=str,
        required=False,
        help="load directory of mdp?",
    )
    parser.add_argument(
        "--real",
        type=str2bool,
        required=True,
        help="use real parameter or not",
    )
    args = parser.parse_args([
        "--solver", "val",
        "--parallel", "true",
        "--num_actions", "2",
        "--sigma", "0",
        "--load", "false",
        "--real", "true"
    ])

    dp = DKC_DP(args)
    dp.set_mdp(args)
    dp.run(args)