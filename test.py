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

    #Get test configuration
    device = torch.device('cuda' if cfg['device']=='cuda' and torch.cuda.is_available() else 'cpu')
    cube_size = cfg['test']['cube_size']
    state_dim, action_dim = get_env_config(cube_size)
    hidden_dim = cfg['model']['hidden_dim']
    deepcube = DeepCube(state_dim, action_dim, hidden_dim).to(device)
    env = make_env(device, cube_size)
    test_model_path = cfg['test']['test_model_path']
    save_file_path = cfg['test']['save_file_path']
    show_scramble_count = cfg['test']['show_scramble_count']
    
    os.makedirs(test_model_path, exist_ok = True)
    os.makedirs(save_file_path, exist_ok = True)

    if mode == 'save':
        time1 = time.time()
        # plot_test_solve_percentage(deepcube, env, cfg, device)
        plot_test_distribution_and_dispersion(deepcube, env, cfg, device)
        print(time.time()-time1)
        pass
    elif mode == 'show': # 화면에 띄우기
        deepcube.load_state_dict(torch.load(test_model_path + '\\7500.pt', map_location = device)['model_state_dict'])
        trial(deepcube, env, cfg, show_scramble_count, seed=None, mask=True, save=False)
        # trial 결과 나왔으니 시뮬레이션 띄우기 
        pass
    else : # show랑 save 둘다 아님
        raise ValueError('You can only choose \"show\" or \"save\" as mode factor')

def trial(model, env, cfg, scramble_count, seed = None, mask=True, save = True):
    """
    Try to solve a cube with given state
    Args:
        model: trained DeepCube model
        env: Cube environment
        cfg: Which contains trial configuration
        scramble_count: scramble count
        seed: seed to apply scrambling cube except None (default to None)
        mask:
        save:
    Returns:
        solve_count: count to solve a cube  
        solve_time: time to solve a cube (sec)
        trial_result: True or False
    """
    max_timesteps = cfg['test']['max_timesteps']
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

def plot_test_solve_percentage(model, env, cfg, device):
    """
    Make plot which shows solve percentage by scramble counts
    Args:
        model: trained DeepCube model
        env: Cube environment
        cfg: Which contains trial configuration
        device:
        masks: 'only' or 'both'
    """

    test_model_path = cfg['test']['test_model_path']
    save_file_path = cfg['test']['save_file_path']
    trial_scramble_count = 30
    trial_cube_count = 50
    model_list = os.listdir(test_model_path)
    masks = cfg['test']['masks']
    if masks == 'only' :
        masks = [True]
    elif masks == 'both':
        masks = [True, False]
    else:
        masks = [False]

    fig, ax = plt.subplots(figsize = (6,6), facecolor = '#c1f1f1')
    ax.set_title('Difficulty vs Solve Percentage')
    ax.set_xlabel('Scramble Distance')
    ax.set_ylabel('Percentage solved(%)')

    for model_name in model_list:
        model.load_state_dict(torch.load(test_model_path + '/' + model_name, map_location = device)['model_state_dict'])

        solve_count_table = np.zeros((trial_scramble_count, trial_cube_count))
        solve_time_table = np.zeros((trial_scramble_count, trial_cube_count))
        trial_result_table = np.zeros((trial_scramble_count, trial_cube_count))
            
        for mask in masks:
            for scramble_count in range(1, trial_scramble_count+1):
                for cube_count in range(trial_cube_count):
                    solve_count, solve_time, trial_result = trial(model, env, cfg, scramble_count, seed=cube_count, mask=mask)
                    
                    solve_count_table[scramble_count-1][cube_count] = solve_count
                    solve_time_table[scramble_count-1][cube_count] = solve_time
                    trial_result_table[scramble_count-1][cube_count] = trial_result

            scramble_distance_list = [i for i in range(1, trial_scramble_count+1)]
            solve_count_list = [trial_result_table[scramble_count-1].mean() * 100 for scramble_count in range(1, trial_scramble_count+1)]
            ax.plot(scramble_distance_list, solve_count_list, '--', label = f'{"Mask"*mask} {model_name[:-3]}')

    ax.legend(loc = 'upper right')
    ax.set_xticks(np.linspace(1, trial_scramble_count+1, 7, endpoint = True))
    ax.set_yticks(np.linspace(0, 100, 6, endpoint = True))
    plt.savefig(f'{save_file_path}/Difficulty vs Solve Percentage.png', dpi = 300)
    # plt.show()

def plot_test_distribution_and_dispersion(model, env, cfg, device):
    """
    Make histogram which shows distribution of solve times and boxplot which shows dispersion of scramble counts 
    Args:
        model: trained DeepCube model
        env: Cube environment
        cfg: Which contains trial configuration
        device:
        masks: 'only' or 'both'
    """

    test_model_path = cfg['test']['test_model_path']
    save_file_path = cfg['test']['save_file_path']
    trial_scramble_count = 1000
    trial_cube_count = 640
    model_list = os.listdir(test_model_path)
    masks = cfg['test']['masks']
    if masks == 'only' :
        masks = [True]
    elif masks == 'both':
        masks = [True, False]
    else:
        masks = [False]

    fig, axes = plt.subplots(2,1, figsize = (6,12), facecolor = '#c1f1f1')
    ax2 = axes[1].twinx()
    axes[0].set_title('Distribution of Solve Time')
    axes[0].set_xlabel('Seconds')
    axes[0].set_ylabel('Number of cubes')
    axes[1].set_title('Dispersion of Scramble Counts')
    axes[1].set_xlabel('Model')
    axes[1].set_ylabel('Number of moves')
    xlabel_list = []

    for idx1, model_name in enumerate(model_list):
        model.load_state_dict(torch.load(test_model_path + '\\' + model_name, map_location = device)['model_state_dict'])

        solve_count_table = np.zeros(trial_cube_count)
        solve_time_table = np.zeros(trial_cube_count)
        trial_result_table = np.zeros(trial_cube_count)
        
        for idx2, mask in enumerate(masks):
            for cube_count in range(trial_cube_count):
                seed = cube_count
                solve_count, solve_time, trial_result = trial(model, env, cfg, trial_scramble_count, seed, mask=mask)
                
                solve_count_table[cube_count] = solve_count
                solve_time_table[cube_count] = solve_time
                trial_result_table[cube_count] = trial_result
            
            solve_time_list_s = solve_time_table[~np.isnan(solve_count_table)]
            axes[0].hist(solve_time_list_s, range=(0, 0.04), bins=40, alpha = 0.9, label = f'{"Mask"*mask} {model_name[:-3]}', histtype='step')
            axes[1].boxplot(solve_count_table[~np.isnan(solve_count_table)], positions=range(idx1*2+idx2+1, idx1*2+idx2+2))
            ax2.axhline(trial_result_table.mean()*100, xmin=(idx1*2+idx2+1+0.5-0.5)/((len(masks)+1) * len(model_list)+1),
                                          xmax=(idx1*2+idx2+1+0.5+0.5)/((len(masks)+1) * len(model_list)+1), color = 'red', linestyle='--')
            xlabel_list.append(f'{"Mask"*mask} {model_name[:-3]}')
    
        axes[0].legend(loc = 'upper right')
        if len(xlabel_list) < 5:  
            axes[1].set_xticklabels(xlabel_list)
        else:
            axes[1].set_xticklabels(range(1, len(xlabel_list)+1))
        ax2.set_yticks(range(0, 101, 10))

    plt.savefig(f'{save_file_path}/Dispersion of scramble count.png', dpi = 300)
    # plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model', type=str, default='', help='Path to pretrained model file')
    args = parser.parse_args()

    with open('./config/config.yaml') as f:
        cfg = yaml.safe_load(f)
    test(cfg, mode = 'save')