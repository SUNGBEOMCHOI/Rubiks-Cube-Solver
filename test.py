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
import matplotlib.colors as colors

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
        plot_test_solve_percentage(deepcube, env, cfg, device) # max 200일 때 750초
        # plot_test_distribution_and_dispersion(deepcube, env, cfg, device)
        print(time.time()-time1)
        pass
    elif mode == 'show': # 화면에 띄우기
        deepcube.load_state_dict(torch.load(test_model_path + '\\model_7500.pt', map_location = device)['model_state_dict'])
        _, _, trial_result, action_list = trial(deepcube, env, cfg, show_scramble_count, seed=None, mask=True, save=False)
        analysis(action_list, trial_result, mode)
        # trial 결과 나왔으니 시뮬레이션 띄우기 
        pass
    else : # show랑 save 둘다 아님
        raise ValueError('You can only choose \"show\" or \"save\" as mode factor')

def analysis(action_list, trial_result, mode):
    """
    Analize its action to classify result
    Args:
        action_list: list of action until finish
        trial_result: True of False
        mode: show(print result) or save(hide result)
    Return:
        result: numpy ndarray[성공, 한방향, 반복, 주기성(n >= 3), 모름]
    """
    if mode == 'show':
        print(action_list)
    result = np.zeros(5)
    if trial_result:
        result[0]+=1
        if mode == 'show':
            print('완성')
    else:
        try:
            for idx in range(6, len(action_list)):
                if action_list[-2:] == action_list[-4:-2] and action_list[-2:] != action_list[-3:-1] and action_list[-6:-4] == action_list[-4:-2]:
                    result[1]+=1
                    if mode == 'show':
                        print('두 상태를 왔다갔다 하는 중')
                    break
                elif action_list[-2:] == action_list[-4:-2] and action_list[-2:] == action_list[-3:-1] and action_list[-3:] == action_list[-6:-3]:
                    result[2]+=1
                    if mode == 'show':
                        print('한 방향으로만 진행 중')
                    break
                elif action_list[-idx:] == action_list[-2*idx:-idx]:
                    result[3]+=1
                    if mode == 'show':
                        print('주기성을 가지고 계속 같은 state로 돌아옴')
                    break
                else:
                    if idx == len(action_list)-1 :
                        result[4]+=1
                        # print(action_list, 'max_timestep>200 / 뜨면 알려주세요', sep = '\n')
                    pass
        except:
            pass
    return result
        

def trial(model, env, cfg, scramble_count, seed = None, mask=True):
    """
    Try to solve a cube with given state
    Args:
        model: trained DeepCube model
        env: Cube environment
        cfg: Which contains trial configuration
        scramble_count: scramble count
        seed: seed to apply scrambling cube except None (default to None)
        mask:
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
            else:
                action = model.get_action(state_tensor)
            action_list.append(action)
            pre_action = action

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
    return solve_count, solve_time, trial_result, action_list

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
    colors_list = list(colors._colors_full_map.values())
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


    cube_size = cfg['test']['cube_size']
    state_dim, action_dim = get_env_config(cube_size)
    size = [[256, 64, 32], [512, 128, 64], [128, 32, 16],[64, 16, 8]]
    result=np.zeros([len(model_list)*len(masks), trial_scramble_count, 5])
    name_list = []
    for idxs, model_name in enumerate(model_list):
        model = DeepCube(state_dim, action_dim, size[idxs]).to(device)
        model.load_state_dict(torch.load(test_model_path + '/' + model_name, map_location = device)['model_state_dict'])

        solve_count_table = np.zeros((trial_scramble_count, trial_cube_count))
        solve_time_table = np.zeros((trial_scramble_count, trial_cube_count))
        trial_result_table = np.zeros((trial_scramble_count, trial_cube_count))
        
        for idx, mask in enumerate(masks):
            name_list.append(f'{"Mask"*mask} {model_name[:-14]}')
            for scramble_count in range(1, trial_scramble_count+1):
                tmp = np.zeros(5)
                for cube_count in range(trial_cube_count):
                    solve_count, solve_time, trial_result, action_list = trial(model, env, cfg, scramble_count, seed=cube_count, mask=mask)
                    tmp += analysis(action_list, trial_result, mode = 'save')

                    solve_count_table[scramble_count-1][cube_count] = solve_count
                    solve_time_table[scramble_count-1][cube_count] = solve_time
                    trial_result_table[scramble_count-1][cube_count] = trial_result
                result[idxs*len(masks)+idx][scramble_count-1] = tmp
            scramble_distance_list = [i for i in range(1, trial_scramble_count+1)]
            solve_count_list = [trial_result_table[scramble_count-1].mean() * 100 for scramble_count in range(1, trial_scramble_count+1)]
            ax.plot(scramble_distance_list, solve_count_list, '--', label = f'{"Mask"*mask} {model_name[:-3]}')

    ax.legend(loc = 'upper right')
    ax.set_xticks(np.linspace(1, trial_scramble_count+1, 7, endpoint = True))
    ax.set_yticks(np.linspace(0, 100, 6, endpoint = True))
    plt.savefig(f'{save_file_path}/Difficulty vs Solve Percentage.png', dpi = 300)
    # plt.show()
    
    for idx in range(len(result[:,0,0])): #모델 별
        fig, ax = plt.subplots(figsize=(12,6), facecolor = '#c1f1f1')
        ax.set_title('Analysis of results')
        ax.set_xlabel('Example counts')
        ax.set_ylabel('Scramble counts')
        for category in range(len(result[0,0])):
            ax.bar(x=range(1, len(result[idx,:,0])+1), height=result[idx,:,category], width=0.7, color = colors_list[category], bottom=result[idx,:,:category].sum(axis=1))
        ax.set_xticks(range(1, trial_scramble_count+1, 5))
        ax.set_yticks(range(0, trial_cube_count+1, 10))
        plt.savefig(f'{save_file_path}/{name_list[idx]} analysis.png', dpi = 300)

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
    cube_size = cfg['test']['cube_size']
    state_dim, action_dim = get_env_config(cube_size)
    size = [[256, 64, 32], [512, 128, 64], [128, 32, 16],[64, 16, 8]]
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
        model = DeepCube(state_dim, action_dim, size[idx1]).to(device)
        model.load_state_dict(torch.load(test_model_path + '\\' + model_name, map_location = device)['model_state_dict'])

        solve_count_table = np.zeros(trial_cube_count)
        solve_time_table = np.zeros(trial_cube_count)
        trial_result_table = np.zeros(trial_cube_count)
        
        for idx2, mask in enumerate(masks):
            for cube_count in range(trial_cube_count):
                seed = cube_count
                solve_count, solve_time, trial_result, _ = trial(model, env, cfg, trial_scramble_count, seed, mask=mask)
                
                solve_count_table[cube_count] = solve_count
                solve_time_table[cube_count] = solve_time
                trial_result_table[cube_count] = trial_result
            
            solve_time_list_s = solve_time_table[~np.isnan(solve_count_table)]
            axes[0].hist(solve_time_list_s, range=(0, 0.04), bins=40, alpha = 0.9, label = f'{"Mask"*mask} {model_name[:-3]}', histtype='step')
            axes[1].boxplot(solve_count_table[~np.isnan(solve_count_table)], positions=range(idx1*2+idx2+1, idx1*2+idx2+2))
            ax2.axhline(trial_result_table.mean()*100, xmin=(idx1*2+idx2+0.5-0.2)/((len(masks)) * len(model_list)),
                                          xmax=(idx1*2+idx2+0.5+0.2)/((len(masks)) * len(model_list)), color = 'red', linestyle='--')
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