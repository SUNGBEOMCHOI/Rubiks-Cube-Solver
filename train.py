import os
from collections import defaultdict

import yaml
import numpy as np
import torch

from model import DeepCube
from env import make_env
from utils import ReplayBuffer, get_env_config, loss_func, optim_func, \
    update_params, plot_progress, plot_valid_hist, save_model

def train(cfg):
    """
    

    """
    ############################
    # Get train configuration  #
    ############################
    device = torch.device('cuda' if cfg['device']=='cuda' and torch.cuda.is_available() else 'cpu')
    
    batch_size = cfg['train']['batch_size']
    epochs = cfg['train']['epochs']
    sample_epoch = cfg['train']['sample_epoch']
    sample_scramble_count = cfg['train']['sample_scramble_count']
    sample_cube_count = cfg['train']['sample_cube_count']
    buffer_size = cfg['train']['buffer_size']
    validation_epoch = cfg['train']['validation_epoch']
    video_path = cfg['train']['video_path']
    model_path = cfg['train']['model_path']
    progress_path = cfg['train']['progress_path']
    cube_size = cfg['env']['cube_size']
    state_dim, action_dim = get_env_config(cube_size=3)

    ############################
    #      Train settings      #
    ############################
    deepcube = DeepCube(state_dim, action_dim).to(device)
    env = make_env(cube_size)

    criterion_list = loss_func()
    optim_list = optim_func()

    replay_buffer = ReplayBuffer(buffer_size, device)
    loss_history = defaultdict(lambda: {'loss':[]})
    valid_history = defaultdict(lambda: {'solve_percentage':[]})

    os.makedirs(video_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(progress_path, exist_ok=True)

    ############################
    #       train model        #
    ############################
    for epoch in range(1, epochs):
        if (epoch-1) % sample_epoch == 0:
            env.get_random_samples(replay_buffer, sample_scramble_count, sample_cube_count)
        loss = update_params(deepcube, replay_buffer, criterion_list, optim_list, batch_size, device)
        loss_history[epoch]['loss'].append(loss)
        if (epoch-1) % validation_epoch == 0:
            validation(deepcube, env, valid_history, epoch, cfg)
            plot_valid_hist(valid_history, save_file_path=progress_path)
            save_model(deepcube, epoch, video_path)
        plot_progress(loss_history, save_file_path=progress_path)

def validation(model, env, valid_history, epoch, cfg):
    """
    Validate model, Solve scrambled cubes with trained model and save video

    Args:
        model: trained DeepCube model
        env: Cube environment
        valid_history: Dictionary to store results
        epoch: Current epoch
        cfg: Which contains validation configuration
    """
    max_timesteps = cfg['validation']['max_timesteps']
    sample_scramble_count = cfg['validation']['sample_scramble_count']
    sample_cube_count = cfg['validation']['sample_cube_count']
    seed = [i for i in range(sample_cube_count)]
    # TODO: 비디오 저장이 가능하도록
    for scramble_count in range(1, sample_scramble_count+1):
        solve_count = 0
        for idx in sample_cube_count:
            state, done = env.reset(seed, scramble_count), False
            for timestep in range(1, max_timesteps+1):
                with torch.no_grad():
                    action = model.get_action(state)
                next_state, reward, done, info = env.step(action)
                if done:
                    solve_count += 1
                    break
                state = next_state
        solve_percentage = (solve_count/sample_cube_count) * 100
        valid_history[epoch]['solve_percentage'].append(solve_percentage)

if __name__ == "__main__":
    with open('./config/config.yaml') as f:
        cfg = yaml.safe_load(f)
    train(cfg)