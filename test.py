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

    hidden_dim = cfg['model']['hidden_dim']
    device = torch.device('cuda' if cfg['device']=='cuda' and torch.cuda.is_available() else 'cpu')
    cube_size = 2 # cfg['test]['cube_size']
    state_dim, action_dim = get_env_config(cube_size)
    deepcube = DeepCube(state_dim, action_dim, hidden_dim).to(device) # load model
    env = make_env(device, cube_size) # make environment

    test_model_path = '/test/model'
    os.makedirs(test_model_path, exist_ok = True)

    if mode == 'save':
        time1 = time.time()
        plot_test_solve_percentage(deepcube, env, cfg)
        plot_test_distribution_and_dispersion(deepcube, env, cfg)
        print(time.time()-time1)
        pass
    elif mode == 'show': # 화면에 띄우기
        scramble_count = 100
        seed = None
        deepcube.load_state_dict(torch.load(test_model_path + '\\7500.pt', map_location = device)['model_state_dict'])
        trial(deepcube, env, cfg, scramble_count, seed, mask=True, save=False)
        # trial 결과 나왔으니 시뮬레이션 띄우기 
        pass
    else : # show랑 save 둘다 아님
        print('You can only choose \"show\" or \"save\" as mode factor')
        raise ValueError

def trial(model, env, cfg, scramble_count, seed = None, mask=True, save = True):
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
    max_timesteps = 200 # max_timesteps 1000이랑 30이랑 정확하게 동일함
    solve_count, solve_time, trial_result = 0, 0, 0
    state, done, pre_action = env.reset(seed, scramble_count), False, None
    start_time = time.time()
    action_list=[]
    for timestep in range(1, max_timesteps+1):
        with torch.no_grad():
            state_tensor = torch.tensor(state).float().detach()
            if mask:
                action = model.get_action(state_tensor, pre_action)
                if save==False:
                    action_list.append(action)
                pre_action = action
            else:
                action = model.get_action(state_tensor)
        next_state, _, done, _ = env.step(action)
        if done:
            solve_count = timestep
            solve_time = time.time() - start_time
            trial_result = 1
            break
        state = next_state
        
        if timestep == max_timesteps:
            solve_count = None
            solve_time = time.time() - start_time
    if save==False:
        print(action_list)
    return solve_count, solve_time, trial_result

def plot_test_solve_percentage(model, env, cfg, save_file_path = './test', test_model_path = './test/model'):
    """
    Make plot which shows solve percentage by scramble counts
    Args:
        model: trained DeepCube model
        env: Cube environment
        cfg: Which contains trial configuration
    """
    os.makedirs(save_file_path, exist_ok = True)
    os.makedirs(test_model_path, exist_ok = True)
    device = 'cpu'
    trial_scramble_count = 30
    trial_cube_count = 50
    model_list = os.listdir(test_model_path)

    fig, ax = plt.subplots(figsize = (6,6), facecolor = '#c1f1f1')

    for model_name in model_list:
        model.load_state_dict(torch.load(test_model_path + '/' + model_name, map_location = device)['model_state_dict'])

        solve_count_table = np.zeros((trial_scramble_count, trial_cube_count))
        solve_time_table = np.zeros((trial_scramble_count, trial_cube_count))
        trial_result_table = np.zeros((trial_scramble_count, trial_cube_count))
            
        for mask in [True, False]:
            for scramble_count in range(1, trial_scramble_count+1): # 움직인 횟수 별로
                for cube_count in range(trial_cube_count): # 1,2,3,,, 50개의 큐브
                    seed = cube_count
                    solve_count, solve_time, trial_result = trial(model, env, cfg, scramble_count, seed, mask=mask)
                    
                    solve_count_table[scramble_count-1][cube_count] = solve_count
                    solve_time_table[scramble_count-1][cube_count] = solve_time
                    trial_result_table[scramble_count-1][cube_count] = trial_result

            
            scramble_distance_list = [i for i in range(1, trial_scramble_count+1)]
            solve_count_list = [trial_result_table[scramble_count-1].mean() * 100 for scramble_count in range(1, trial_scramble_count+1)]
            ax.plot(scramble_distance_list, solve_count_list, '--', label = f'{"Mask"*mask} {model_name[:-3]}')
            # print(solve_count_list, len(solve_count_list), sep='\n')

    ax.set_title('Difficulty vs Solve Percentage')
    ax.set_xlabel('Scramble Distance')
    ax.set_ylabel('Percentage solved')
    ax.legend(loc = 'upper right')
    ax.set_xticks(np.linspace(1, 31, 7, endpoint = True))
    ax.set_yticks(np.linspace(0, 100, 6, endpoint = True))
    plt.savefig(f'{save_file_path}/Difficulty vs Solve Percentage.png', dpi = 300)
    plt.show()

def plot_test_distribution_and_dispersion(model, env, cfg, save_file_path = './test', test_model_path = './test/model'):
    """
    Make histogram which shows distribution of solve times and boxplot which shows dispersion of scramble counts 
    Args:
        model: trained DeepCube model
        env: Cube environment
        cfg: Which contains trial configuration
    """

    os.makedirs(save_file_path, exist_ok = True)
    os.makedirs(test_model_path, exist_ok = True)
    device = 'cpu'
    trial_scramble_count = 1000
    trial_cube_count = 640
    model_list = os.listdir(test_model_path)

    fig, axes = plt.subplots(2,1, figsize = (6,12), facecolor = '#c1f1f1')
    axes[0].set_title('Distribution of Solve Time')
    axes[0].set_xlabel('Seconds')
    axes[0].set_ylabel('Number of cubes')
    axes[1].set_title('Dispersion of Scramble Counts')
    axes[1].set_xlabel('Model')
    axes[1].set_ylabel('Number of moves')
    # axes[1].set_ylim(0,25)

    for idx1, model_name in enumerate(model_list):
        model.load_state_dict(torch.load(test_model_path + '\\' + model_name, map_location = device)['model_state_dict'])

        solve_count_table = np.zeros(trial_cube_count)
        solve_time_table = np.zeros(trial_cube_count)
        trial_result_table = np.zeros(trial_cube_count)
        
        for idx2, mask in enumerate([True, False]):
            for cube_count in range(trial_cube_count): # 1,2,3,,, 640개의 큐브
                seed = cube_count
                solve_count, solve_time, trial_result = trial(model, env, cfg, trial_scramble_count, seed, mask=mask)
                
                solve_count_table[cube_count] = solve_count
                solve_time_table[cube_count] = solve_time
                trial_result_table[cube_count] = trial_result
            
            solve_time_list_s = solve_time_table[~np.isnan(solve_count_table)]
            solve_time_list_f = solve_time_table[np.isnan(solve_count_table)]
            axes[0].hist(solve_time_list_s, range=(0, 0.04), bins=40, alpha = 0.9, label = f'{"Mask"*mask} {model_name[:-3]}', histtype='step')
            # axes[0].hist([solve_time_list_s, solve_time_list_f], bins = 100, label = ['Success', 'Failure'], color = ['blue', 'red'])
            axes[1].boxplot(solve_count_table[~np.isnan(solve_count_table)], positions=range(idx1*2+idx2+1, idx1*2+idx2+2))
            # print(solve_count_table[~np.isnan(solve_count_table)], solve_count_table[~np.isnan(solve_count_table)].shape, sep= '\n')
    
        axes[0].legend(loc = 'upper right')
        axes[1].set_xticks([idx for idx in range(1, (idx2+1)*len(model_list)+1)])
        # axes[1].set_xticklabels([f'{"Mask"*mask} {model_name[:-3]}' for model_name in model_list])
        axes[1].set_xticklabels(['Mask 2600', '2600', 'Mask 7500', '7500'])

    # # plt.figure(figsize = (6,6))
    # # plt.boxplot(solve_count_table[~np.isnan(solve_count_table)])
    # plt.title('Dispersion of Scramble Counts')
    # plt.xlabel('Model')
    # plt.ylabel('Number of moves')
    # plt.xticks([1], ['model1'])
    plt.savefig(f'{save_file_path}/Dispersion of scramble count.png', dpi = 300)
    # plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model', type=str, default='', help='Path to pretrained model file')
    args = parser.parse_args()

    with open('./config/config.yaml') as f:
        cfg = yaml.safe_load(f)
    test(cfg, mode = 'save')