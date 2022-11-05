import argparse
import matplotlib.patches as mpatches
import winsound as sd

import torch
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import yaml
from model import DeepCube
from env import make_env
from utils import get_env_config
from mcts import MCTS

# mcts, mask, size

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
    # hidden_dim = cfg['model']['hidden_dim']
    test_model_path = cfg['test']['test_model_path']
    save_file_path = cfg['test']['save_file_path']
    show_scramble_count = cfg['test']['show_scramble_count']
    
    os.makedirs(test_model_path, exist_ok = True)
    os.makedirs(save_file_path, exist_ok = True)
    env = make_env(device, cube_size)

    if mode == 'save':
        time1 = time.time()
        # plot_test_solve_percentage(env, cfg, device) # max 200일 때 750초
        plot_test_distribution_and_dispersion(env, cfg, device) #max 200일 때 600초
        print(time.time()-time1)
        pass
    elif mode == 'show': # 화면에 띄우기
        model_name = os.listdir(test_model_path)[0]
        deepcube = model_load(test_model_path, model_name, device, state_dim, action_dim)
        time1 = time.time()
        _, _, trial_result, action_list = trial(deepcube, env, cfg, show_scramble_count, seed=None, mask=False, mcts_=False)
        analysis(action_list, trial_result, mode)
        print(time.time()-time1)
        # trial 결과 나왔으니 시뮬레이션 띄우기 
        pass
    else : # show랑 save 둘다 아님
        raise ValueError('You can only choose \"show\" or \"save\" as mode factor')

def model_load(test_model_path, model_name, device, state_dim, action_dim):
    model_state = torch.load(f'{test_model_path}\{model_name}', map_location = device)
    hidden_dim = [model_state['optimizer_state_dict']['state'][x]['exp_avg'].size(0) for x in [1, 3, 5]]
    model = DeepCube(state_dim, action_dim, hidden_dim).to(device)
    model.load_state_dict(model_state['model_state_dict'])
    return model

def analysis(action_list, trial_result, mode):
    """
    Analize its action to classify result
    Args:
        action_list: list of action until finish
        trial_result: True of False
        mode: show(print result) or save(hide result)
    Return:
        result: numpy ndarray[성공, 한방향, 반복, 주기(n >= 3), 예외]
    """
    result = np.zeros(5, dtype='int64')
    if trial_result:
        result[0]+=1
    else:        
        for idx in range(0, len(action_list)):
            if action_list[-2:] == action_list[-4:-2] and action_list[-2:] != action_list[-3:-1] and action_list[-6:-4] == action_list[-4:-2]:
                result[1]+=1
                break
            elif action_list[-2:] == action_list[-4:-2] and action_list[-2:] == action_list[-3:-1] and action_list[-3:] == action_list[-6:-3]:
                result[2]+=1
                break
            else:
                try:
                    for x in range(4,14):
                        if action_list[-(x+idx):-idx] == action_list[-(2*x+idx):-(idx+x)]:
                            result[3]+=1
                            break
                except:
                    pass
            if idx == len(action_list)-1 :
                result[4]+=1
                pass
        if mode == 'show':
            output = ['완성', "like U U' U U'", "like U U U U", '주기(n>3)', '예외']
            for x in range(5):
                if result[x] == 1:
                    print(output[x], action_list, sep='\n')
                else:
                    pass
    return result

def trial(model, env, cfg, scramble_count, seed = None, mask=False, mcts_=False):
    """
    Try to solve a cube with given state
    Args:
        model: trained DeepCube model
        env: Cube environment
        cfg: Which contains trial configuration
        scramble_count: scramble count
        seed: seed to apply scrambling cube except None (default to None)
        mask: True or False
        mcts_: True or False

    Returns:
        solve_scramble_count: count to solve a cube  
        solve_time_time: time to solve a cube (sec)
        trial_result: True or False
    """
    max_timesteps = cfg['test']['max_timesteps']
    solve_scramble_count, solve_time_time, trial_result = 0, 0, 0
    state, done, pre_action = env.reset(seed, scramble_count), False, None
    start_time = time.time()
    if mcts_:
        mcts = MCTS(model, cfg)
    action_list=[]
    for timestep in range(1, max_timesteps+1):
        if mcts_ == False:
            with torch.no_grad():
                state_tensor = torch.tensor(state).float().detach()
                if mask:
                    action = model.get_action(state_tensor, pre_action)
                    pre_action = action
                else:
                    action = model.get_action(state_tensor)
                action_list.append(action)
            next_state, _, done, _ = env.step(action)
        else:
            with torch.no_grad():
                action = mcts.getActionProb(state, env, temp=0)
            action_list.append(action.index(1))
            next_state, _, done, _ = env.step(action.index(max(action)))
        if done:
            solve_scramble_count = timestep
            solve_time_time = time.time() - start_time
            trial_result = 1
            break
        state = next_state
        if timestep == max_timesteps:
            solve_scramble_count = None
            solve_time_time = time.time() - start_time
    return solve_scramble_count, solve_time_time, trial_result, action_list

def data_for_plot(model, env, cfg, trial_scramble_count, trial_cube_count, mask, mcts_, iter):
    if iter:
        result1 = np.zeros((3, 1, trial_scramble_count, trial_cube_count))
        result2 = np.zeros((5, 1, trial_scramble_count, trial_cube_count), dtype='int64')
        for scramble_count in range(1, trial_scramble_count+1):
            for cube_count in range(trial_cube_count):
                solve_scarmable_count, solve_time_time, trial_result, action_list = trial(model, env, cfg, scramble_count, seed = cube_count, mask=mask, mcts_=mcts_)
                result2[:,0,scramble_count-1,cube_count] += analysis(action_list, trial_result, mode='save')
                result1[:,0,scramble_count-1,cube_count] = np.array([solve_scarmable_count, solve_time_time, trial_result])
    else:
        result1 = np.zeros((3, 1, 1, trial_cube_count))
        result2 = ''
        for cube_count in range(trial_cube_count):
            solve_scarmable_count, solve_time_time, trial_result, action_list = trial(model, env, cfg, trial_scramble_count, seed = cube_count, mask=mask, mcts_=mcts_)
            result1[:,0,0,cube_count] = np.array([solve_scarmable_count, solve_time_time, trial_result])
    return result1, result2

def set_option(model_list, masks, mcts_):
    result = []
    for x in model_list:
        for y in masks:
            for z in mcts_:
                result.append([x, y, z])
    return result

def plot_solve_ratio(array, options, colors_list, save_file_path):
    fig, ax = plt.subplots(figsize = (6,6), facecolor = '#c1f1f1')
    ax.set_title('Difficulty vs Solve Percentage')
    ax.set_xlabel('Scramble Distance')
    ax.set_ylabel('Percentage solved(%)')
    x=range(1, len(array[0,0,:,0])+1)
    for idx, option in enumerate(options):
        y=[array[2, idx, x-1, :,].mean()*100 for x in range(1, len(array[0,0,:,0])+1)]
        ax.plot(x, y, '--', label = f"{'MCTS'*option[2]}{'Mask'*option[1]}{option[0][:-14]}", color = colors_list[idx])
    ax.legend(loc = 'upper right')
    ax.set_xticks(np.linspace(1, len(array[0,0,:,0])+1, 7, endpoint = True))
    ax.set_yticks(np.linspace(0, 100, 6, endpoint = True))
    plt.savefig(f'{save_file_path}/Difficulty vs Solve Percentage.png', dpi = 300)

def plot_analysis(array, options, colors_list, save_file_path):
    list_legend = ['success', "like U U' U U'", "like U U U U", 'periodic', 'exception']
    for idx, option in enumerate(options): #모델 별
        fig, ax = plt.subplots(figsize=(12,6), facecolor = '#c1f1f1')
        ax.set_title('Analysis of results')
        ax.set_ylabel('Example counts')
        ax.set_xlabel('Scramble counts')
        for category in range(len(array[:,0,0,0])):
            x=range(1, len(array[0,0,:,0])+1)
            height=array[category,idx,:,:].sum(axis=1)
            bottom=array[:category,idx,:,:].sum(axis=0).sum(axis=1)
            ax.bar(x, height, width=0.7, color = colors_list[category], bottom=bottom)
        ax.set_xticks(range(1, len(array[0,0,:,0])+1, 5))
        ax.set_yticks(range(0, len(array[0,0,0,:])+6, 5))
        plt.legend(handles=[mpatches.Patch(color=colors_list[x], label=list_legend[x]) for x in range(5)],
                            loc = 'upper center', mode = "expand", ncol=5)
        plt.savefig(f"{save_file_path}/{'MCTS'*option[2]}{'Mask'*option[1]}{option[0][:-14]} analysis.png", dpi = 300)

def plot_ditribution_dispersion(array, options, colors_list, save_file_path):
    fig, axes = plt.subplots(2,1, figsize = (6,12), facecolor = '#c1f1f1')
    ax2 = axes[1].twinx()
    axes[0].set_title('Distribution of Solve Time')
    axes[0].set_xlabel('Seconds')
    axes[0].set_ylabel('Number of cubes')
    axes[1].set_title('Dispersion of Scramble Counts')
    axes[1].set_xlabel('Model')
    axes[1].set_ylabel('Number of moves')
    #array [3, idx, 1, 640]
    xlabel_list=[]
    for idx, option in enumerate(options):
        axes[0].hist(array[1,idx,0,~np.isnan(array[0,idx,0,:])], range=(0,0.04), bins=40, alpha=0.9, color=colors_list[idx],
                     label=f"{'MCTS'*option[2]}{'Mask'*option[1]}{option[0][:-14]}", histtype='step')
        axes[1].boxplot(array[0,idx,0,~np.isnan(array[0,idx,0,:])], positions=[idx])
        ax2.axhline(y = array[2,idx,0,:].mean()*100, xmin=idx/len(options), xmax=(idx+1)/len(options), color='red', linestyle='--')
        xlabel_list.append(f"{'MCTS'*option[2]}{'Mask'*option[1]}{option[0][:-14]}")
    axes[0].legend(loc='upper right')
    if len(xlabel_list) < 5:
        axes[1].set_xticklabels(xlabel_list)
        ax2.set_yticks(range(0, 101, 10))
    plt.savefig(f'{save_file_path}/Dispersion of scramble count.png', dpi = 300)

def plot_test_solve_percentage(env, cfg, device):
    """
    Make plot which shows solve percentage by scramble counts
    Args:
        env: Cube environment
        cfg: Which contains trial configuration
        device:
        masks: 'only' or 'both'
    """
    trial_scramble_count = 30
    trial_cube_count = 50
    test_model_path = cfg['test']['test_model_path']
    save_file_path = cfg['test']['save_file_path']
    model_list = os.listdir(test_model_path)
    colors_list = list(colors._colors_full_map.values())
    dicts = {'only':[True],'both':[True, False],'x':[False]}
    masks = dicts[cfg['test']['masks']]
    mcts_ = dicts[cfg['test']['mcts_']]
    options = set_option(model_list, masks, mcts_)
    cube_size = cfg['test']['cube_size']
    state_dim, action_dim = get_env_config(cube_size)

    result1 = np.zeros((3, len(options), trial_scramble_count, trial_cube_count))
    result2 = np.zeros((5, len(options), trial_scramble_count, trial_cube_count))
    for idx, option in enumerate(options):
        deepcube = model_load(test_model_path, option[0], device, state_dim, action_dim)
        result1[:,idx:idx+1,:,:], result2[:,idx:idx+1,:,:] = data_for_plot(deepcube, env, cfg, trial_scramble_count, trial_cube_count, mask=option[1], mcts_=option[2], iter=True)

    plot_solve_ratio(result1, options, colors_list, save_file_path)
    plot_analysis(result2, options, colors_list, save_file_path)

def plot_test_distribution_and_dispersion(env, cfg, device):
    """
    Make histogram which shows distribution of solve times and boxplot which shows dispersion of scramble counts 
    Args:
        env: Cube environment
        cfg: Which contains trial configuration
        device:
    """
    trial_scramble_count = 1000
    trial_cube_count = 640
    test_model_path = cfg['test']['test_model_path']
    save_file_path = cfg['test']['save_file_path']
    model_list = os.listdir(test_model_path)
    colors_list = list(colors._colors_full_map.values())
    dicts = {'only':[True],'both':[True, False],'x':[False]}
    masks = dicts[cfg['test']['masks']]
    mcts_ = dicts[cfg['test']['mcts_']]
    options = set_option(model_list, masks, mcts_)
    cube_size = cfg['test']['cube_size']
    state_dim, action_dim = get_env_config(cube_size)

    result1 = np.zeros((3, len(options), trial_scramble_count, trial_cube_count))
    for idx, option in enumerate(options):
        deepcube = model_load(test_model_path, option[0], device, state_dim, action_dim)
        result1[:,idx:idx+1,0:1,:], _ = data_for_plot(deepcube, env, cfg, trial_scramble_count, trial_cube_count, mask=option[1], mcts_=option[2], iter=False)
    plot_ditribution_dispersion(result1, options, colors_list, save_file_path)

def beepsound():
    fr = 2000    # range : 37 ~ 32767
    du = 1000     # 1000 ms ==1second
    sd.Beep(fr, du) # winsound.Beep(frequency, duration)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model', type=str, default='', help='Path to pretrained model file')
    args = parser.parse_args()

    with open('./config/config.yaml') as f:
        cfg = yaml.safe_load(f)
    test(cfg, mode = 'save')
    beepsound()
