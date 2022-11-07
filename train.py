import os
import time
import argparse
from collections import defaultdict
from multiprocessing.managers import BaseManager, DictProxy

import yaml
from tqdm import tqdm
import numpy as np
import torch
import torch.multiprocessing as mp

from model import DeepCube
from env import make_env
from utils import ReplayBuffer, get_env_config, loss_func, optim_func, scheduler_func,\
    update_params, plot_progress, plot_valid_hist, save_model

def train(cfg, args):
    """
    Train model
    """
    ############################
    # Get train configuration  #
    ############################
    device = torch.device('cuda' if cfg['device']=='cuda' and torch.cuda.is_available() else 'cpu')
    batch_size = cfg['train']['batch_size']
    sample_size = cfg['train']['sample_size']
    learning_rate = cfg['train']['learning_rate']
    epochs = cfg['train']['epochs']
    sample_epoch = cfg['train']['sample_epoch']
    sample_scramble_count = cfg['train']['sample_scramble_count']
    sample_cube_count = cfg['train']['sample_cube_count']
    buffer_size = cfg['train']['buffer_size']
    temperature = cfg['train']['temperature']
    validation_epoch = cfg['train']['validation_epoch']
    num_processes = cfg['train']['num_processes']
    video_path = cfg['train']['video_path']
    model_path = cfg['train']['model_path']
    progress_path = cfg['train']['progress_path']
    cube_size = cfg['env']['cube_size']
    state_dim, action_dim = get_env_config(cube_size)
    hidden_dim = cfg['model']['hidden_dim']

    ############################
    #      Train settings      #
    ############################
    deepcube = DeepCube(state_dim, action_dim, hidden_dim).to(device)
    if num_processes: # Use multi process
        deepcube.share_memory() # global model
        optimizer = optim_func(deepcube, learning_rate)
        optimizer.share_memory()
        lr_scheduler = scheduler_func(optimizer)
        
        BaseManager.register('defaultdict', defaultdict, DictProxy)
        mgr = BaseManager()
        mgr.start()
        loss_history = mgr.defaultdict(dict)
        valid_history = mgr.defaultdict(dict)
    else:
        env = make_env(device, cube_size)
        start_epoch = 1

        criterion_list = loss_func()
        optimizer = optim_func(deepcube, learning_rate)
        lr_scheduler = scheduler_func(optimizer)
        lr_scheduler = None

        replay_buffer = ReplayBuffer(buffer_size, sample_size)
        loss_history = defaultdict(lambda: {'loss':[]})
        valid_history = defaultdict(lambda: {'solve_percentage':[]})

    os.makedirs(video_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(progress_path, exist_ok=True)

    if args.resume:
        checkpoint = torch.load(args.path)
        start_epoch = checkpoint['epoch']+1
        deepcube.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    
    ############################
    #       train model        #
    ############################
    if num_processes: # Use multi process
        worker_epochs_list = [epochs // num_processes for _ in range(num_processes)]
        for i in range(epochs % num_processes):
            worker_epochs_list[i] += 1
        workers = [mp.Process(target=single_train, args=(worker_idx, worker_epochs_list[worker_idx-1], deepcube, optimizer, lr_scheduler, valid_history, loss_history, cfg))\
                     for worker_idx in range(1, num_processes+1)]
        [w.start() for w in workers]
        [w.join() for w in workers]

    else: # if num_processes == 0, then train with single machine
        for epoch in tqdm(range(start_epoch, epochs+1)):
            a = time.time()
            if (epoch-1) % sample_epoch == 0: # replay buffer에 random sample저장
                env.get_random_samples(replay_buffer, deepcube, sample_scramble_count, sample_cube_count)
            loss = update_params(deepcube, replay_buffer, criterion_list, optimizer, batch_size, device, temperature)
            loss_history[epoch]['loss'].append(loss)
            if epoch % validation_epoch == 0:
                validation(deepcube, env, valid_history, epoch, device, cfg)
                plot_valid_hist(valid_history, save_file_path=progress_path, validation_epoch=validation_epoch)
                save_model(deepcube, epoch, optimizer, lr_scheduler, model_path)
                plot_progress(loss_history, save_file_path=progress_path)
            print(f'{epoch} : Time {time.time()-a}')
            # lr_scheduler.step()

def single_train(worker_idx, local_epoch_max, global_deepcube, optimizer, lr_scheduler, valid_history, loss_history, cfg):
    """
    Function for train on single process

    Args:
        worker_idx: Process index
        local_epoch_max: Train epoch on single process
        global_deepcube: Shared global train model
        optimizer: Torch optimizer for global deepcube parameters
        lr_scheduler: Leaning rate scheduler (Not available now)
        valid_history: Dictionary for saving validation result
        loss_history: Dictionary for saving loss history
        cfg: config data from yaml file    
    """
    device = torch.device(f'cpu:{worker_idx}')
    torch.set_num_threads(1)
    batch_size = cfg['train']['batch_size']
    sample_size = cfg['train']['sample_size']
    epochs = cfg['train']['epochs']
    sample_epoch = cfg['train']['sample_epoch']
    sample_scramble_count = cfg['train']['sample_scramble_count']
    sample_cube_count = cfg['train']['sample_cube_count']
    buffer_size = cfg['train']['buffer_size']
    temperature = cfg['train']['temperature']
    validation_epoch = cfg['train']['validation_epoch']
    num_processes = cfg['train']['num_processes']
    video_path = cfg['train']['video_path']
    model_path = cfg['train']['model_path']
    progress_path = cfg['train']['progress_path']
    cube_size = cfg['env']['cube_size']
    state_dim, action_dim = get_env_config(cube_size)
    hidden_dim = cfg['model']['hidden_dim']

    global_deepcube = global_deepcube
    deepcube = DeepCube(state_dim, action_dim, hidden_dim).to(device)
    deepcube.load_state_dict(global_deepcube.state_dict())
    env = make_env(device, cube_size)
    local_epoch = 0

    optimizer = optimizer
    criterion_list = loss_func()
    lr_scheduler = lr_scheduler

    replay_buffer = ReplayBuffer(buffer_size, sample_size)
    valid_history = valid_history
    loss_history = loss_history

    while local_epoch < local_epoch_max:
        local_epoch += 1
        if (local_epoch-1) % sample_epoch == 0:
            env.get_random_samples(replay_buffer, deepcube, sample_scramble_count, sample_cube_count)
        deepcube.load_state_dict(global_deepcube.state_dict())
        loss = update_params(deepcube, replay_buffer, criterion_list, optimizer, batch_size, device, temperature, global_deepcube)
        global_epoch = len(loss_history)+1
        loss_history[global_epoch] = {'loss':[loss]}
        print(f"Train progress : {global_epoch} / {epochs} Loss : {loss}")
        if global_epoch % validation_epoch == 0:
            plot_progress(loss_history, save_file_path=progress_path)
            validation(global_deepcube, env, valid_history, global_epoch, device, cfg)
            plot_valid_hist(valid_history, save_file_path=progress_path, validation_epoch=validation_epoch)
            save_model(global_deepcube, global_epoch, optimizer, lr_scheduler, model_path)
        # lr_scheduler.step() # You can run this line for using learning rate scheduler

def validation(model, env, valid_history, epoch, device, cfg):
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
    seed = [i*10 for i in range(sample_cube_count)]
    solve_percentage_list = []
    # TODO: 비디오 저장이 가능하도록
    for scramble_count in range(1, sample_scramble_count+1):
        solve_count = 0
        for idx in range(1, sample_cube_count+1):
            if idx == sample_cube_count and scramble_count==sample_scramble_count: # 마지막 state
                # env.render()
                pass
            state, done = env.reset(seed[idx-1], scramble_count), False
            for timestep in range(1, max_timesteps+1):
                with torch.no_grad():
                    state_tensor = torch.tensor(state).float().to(device).detach()
                    action = model.get_action(state_tensor)
                next_state, reward, done, info = env.step(action)
                if done:
                    solve_count += 1
                    break
                state = next_state
            if idx == sample_cube_count and scramble_count==sample_scramble_count: # 마지막 state render종료
                # env.save_video(cube_size = env.cube_size, scramble_count = scramble_count, sample_cube_count = sample_cube_count, video_path = video_path)
                # env.close_render()
                pass
        solve_percentage = (solve_count/sample_cube_count) * 100
        solve_percentage_list.append(solve_percentage)
        # valid_history[epoch]['solve_percentage'].append(solve_percentage)
    valid_history[epoch] = {'solve_percentage':solve_percentage_list}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default='', help='Path to pretrained model file')
    args = parser.parse_args()

    with open('./config/config.yaml') as f:
        cfg = yaml.safe_load(f)
    train(cfg, args)