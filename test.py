import argparse
from collections import defaultdict
import torch
import os

import yaml

from model import DeepCube
from env import make_env
from utils import get_env_config

def test(cfg, mode = 'show'):
    """
    Test our model
    Args:
        cfg: Which contains validation configuration
        mode: Either viewing directly or saving videos
    """

    device = torch.device('cuda' if cfg['device']=='cuda' and torch.cuda.is_available() else 'cpu')
    cube_size = 3 # cfg['test]['cube_size']
    video_path = './video/test' # cfg['test]['video_path']
    model_path = './pretrained'# cfg['test]['model_path']
    progress_path = '' # cfg['test]['progress_path']

    epochs = 1000 # cfg['test]['epochs']
    state_dim, action_dim = get_env_config(cube_size)

    test_history = defaultdict(lambda: {'solve_percentage':[]})

    deepcube = DeepCube(state_dim, action_dim).to(device) # load model
    env = make_env(cube_size) # make environment
    start_epoch = 1

    os.makedirs(video_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(progress_path, exist_ok=True)

    for epoch in range(start_epoch, epochs+1):
        trial(deepcube, env, test_history, epoch, cfg)
        plot_test_hist(test_history, save_file_path=progress_path)
        if mode == 'show': # View directly
            pass
        else : # Save videos
            pass
    pass

progress_path = ''
def plot_test_hist(test_history, save_file_path=progress_path):
    """
    Make histogram of test results
    Args:
        test_history: Dictionary to store results
        save_file_path: path which saves videos
    """
    pass

def trial(model, env, test_history, epoch, cfg):
    """
    Try to solve scrambled cubes with trained model and save video
    Args:
        model: trained DeepCube model
        env: Cube environment
        test_history: Dictionary to store results
        epoch: Current epoch
        cfg: Which contains validation configuration
    """
    max_timesteps = 1000
    test_scramble_count = 1000
    test_cube_count = 640
    seed = [i for i in range(test_cube_count)]
    # TODO: 비디오 저장이 가능하도록
    for scramble_count in range(1, test_scramble_count+1):
        solve_count = 0
        for idx in range(1, test_cube_count+1):
            if idx == test_cube_count and scramble_count == test_scramble_count: # 마지막 state
                # env.render()
                pass
            state, done = env.reset(seed[idx-1], scramble_count), False
            for timestep in range(1, max_timesteps+1):
                with torch.no_grad():
                    state_tensor = torch.tensor(state).float().detach()
                    action = model.get_action(state_tensor)
                next_state, reward, done, info = env.step(action)
                if done:
                    solve_count += 1
                    break
                state = next_state
            if idx == test_cube_count and scramble_count==test_scramble_count: # 마지막 state render종료
                # env.close_render()
                pass
        solve_percentage = (solve_count/test_cube_count) * 100
        test_history[epoch]['solve_percentage'].append(solve_percentage)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model', type=str, default='', help='Path to pretrained model file')
    args = parser.parse_args()

    with open('./config/config.yaml') as f:
        cfg = yaml.safe_load(f)
    test(cfg)