import argparse

import torch
import os
import time
import numpy as np
import matplotlib.pyplot as plt

import yaml

from model import DeepCube
from env import make_env
from utils import get_env_config

def test(cfg, mode = 'show'):
    """
    Test our model

    Args:
        cfg: Which contains trial configuration
        mode: show(directly pop up video) or save(save test result graphs)
    """
    device = torch.device('cuda' if cfg['device']=='cuda' and torch.cuda.is_available() else 'cpu')
    cube_size = 3 # cfg['test]['cube_size']
    state_dim, action_dim = get_env_config(cube_size)
    deepcube = DeepCube(state_dim, action_dim).to(device) # load model
    env = make_env(cube_size) # make environment

    if mode == 'save':
        plot_test_solve_percentage(deepcube, env)
        plot_test_time_distribution(deepcube, env)
    elif mode == 'show': # 화면에 띄우기
        scramble_count = 100
        seed = None
        trial(deepcube, env, cfg, scramble_count, seed)
        # trial 결과 나왔으니 시뮬레이션 띄우기
        pass
    else : # show랑 save 둘다 아님
        print('You can only choose \"show\" or \"save\" as mode factor')
        raise ValueError

def trial(model, env, cfg, scramble_count, seed = None):
    """
    Try to solve a cube with given state

    Args:
        model: trained DeepCube model
        env: Cube environment
        cfg: Which contains trial configuration
        scramble_count: scramble count
        seed: seed to apply scrambling cube except None (default to None)
    
    Returns:
        solve_count: count to solve a cube  
        solve_time: time to solve a cube (sec)
        trial_result: True or False
    """
    max_timesteps = 1000

    state, done = env.reset(seed, scramble_count), False
    start_time = time.time()
    for timestep in range(1, max_timesteps+1):
        with torch.no_grad():
            state_tensor = torch.tensor(state).float().detach()
            action = model.get_action(state_tensor)
        next_state, reward, done, info = env.step(action) # next_state, _, done, info = env.step(action)
        if done:
            solve_count = timestep
            solve_time = time.time() - start_time
            trial_result = 1
            break
        state = next_state
        if timestep == max_timesteps + 1:
            solve_count = timestep
            solve_time = time.time() - start_time
            trial_result = 0

    return solve_count, solve_time, trial_result

def plot_test_solve_percentage(model, env, cfg):
    """
    Make plot which shows solve percentage by scramble counts

    Args:
        model: trained DeepCube model
        env: Cube environment
        cfg: Which contains trial configuration
    """
    save_file_path = './video'
    os.makedirs(save_file_path, exist_ok = True)
    trial_scramble_count = 30
    trial_cube_count = 50

    solve_count_table = np.zeros((trial_scramble_count, trial_cube_count))
    solve_time_table = np.zeros((trial_scramble_count, trial_cube_count))
    trial_result_table = np.zeros((trial_scramble_count, trial_cube_count))
    
    for distance in trial_scramble_count: # 움직인 횟수 별로
        for scramble_count in trial_cube_count: # 1,2,3,,, 50개의 큐브
            seed = scramble_count
            solve_count, solve_time, trial_result = trial(model, env, cfg, trial_scramble_count, seed)
            
            solve_count_table[distance][scramble_count] = solve_count
            solve_time_table[distance][scramble_count] = solve_time
            trial_result_table[distance][scramble_count] = trial_result

    scramble_distance_list = [i for i in range(trial_scramble_count)]
    solve_count_list = [trial_result_table[distance].mean() * 100 for distance in range(trial_scramble_count)]
    plt.plot(scramble_distance_list, solve_count_list, 'r--')
    plt.title('Difficulty vs Solve Percentage')
    plt.xlabel('Scramble Distance')
    plt.ylabel('Percentage solved')
    plt.legend()
    plt.xticks(np.linspace(0, 30, 5, endpoint = True))
    plt.yticks(np.linspace(0, 100, 5, endpoint = True))
    plt.savefig(f'{save_file_path}/Difficulty vs Solve Percentage.png')

def plot_test_time_distribution(model, env, cfg):
    """
    Make histogram which shows distribution of solve times

    Args:
        model: trained DeepCube model
        env: Cube environment
        cfg: Which contains trial configuration
    """
    save_file_path = './video'
    os.makedirs(save_file_path, exist_ok = True)
    trial_scramble_count = 1000
    trial_cube_count = 640

    solve_count_table = np.zeros(trial_cube_count)
    solve_time_table = np.zeros(trial_cube_count)
    trial_result_table = np.zeros(trial_cube_count)
    
    for cube_count in trial_cube_count: # 1,2,3,,, 640개의 큐브
        seed = cube_count
        solve_count, solve_time, trial_result = trial(model, env, cfg, trial_scramble_count, seed)
        
        solve_count_table[cube_count] = solve_count
        solve_time_table[cube_count] = solve_time
        trial_result_table[cube_count] = trial_result

    solve_time_list = trial_result_table
    plt.hist(solve_time_list, bins = 30, color = 'red')
    plt.axvline(solve_time_list.mean(), color = 'grey', linestyle = '--', label = 'median')
    plt.title('Distribution of Solve Time')
    plt.xlabel('Minutes')
    plt.ylabel('Number of cubes')
    plt.legend()
    plt.xticks(np.linspace(0, 60, 6, endpoint = True))
    plt.yticks(np.linspace(0, 100, 5, endpoint = True))
    plt.savefig(f'{save_file_path}/Distribution of Solve Time.png')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model', type=str, default='', help='Path to pretrained model file')
    args = parser.parse_args()

    with open('./config/config.yaml') as f:
        cfg = yaml.safe_load(f)
    test(cfg)