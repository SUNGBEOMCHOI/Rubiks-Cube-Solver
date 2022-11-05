import os
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
    env = make_env(device, cube_size)
    start_epoch = 1

    criterion_list = loss_func()
    optimizer = optim_func(deepcube, learning_rate)
    lr_scheduler = scheduler_func(optimizer)

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
    for epoch in tqdm(range(start_epoch, epochs+1)):
        if (epoch-1) % sample_epoch == 0: # replay buffer에 random sample저장
            env.get_random_samples(replay_buffer, deepcube, sample_scramble_count, sample_cube_count, temperature)
        loss = update_params(deepcube, replay_buffer, criterion_list, optimizer, batch_size, device, temperature)
        loss_history[epoch]['loss'].append(loss)
        if epoch % validation_epoch == 0:
            validation(deepcube, env, valid_history, epoch, device, cfg)
            plot_valid_hist(valid_history, save_file_path=progress_path, validation_epoch=validation_epoch)
            save_model(deepcube, epoch, optimizer, lr_scheduler, model_path)
            plot_progress(loss_history, save_file_path=progress_path)
        lr_scheduler.step()

def multi_train(cfg, args):
    device = torch.device('cuda' if cfg['device']=='cuda' and torch.cuda.is_available() else 'cpu')
    learning_rate = cfg['train']['learning_rate']
    num_processes = cfg['train']['num_processes']
    video_path = cfg['train']['video_path']
    model_path = cfg['train']['model_path']
    progress_path = cfg['train']['progress_path']
    cube_size = cfg['env']['cube_size']
    state_dim, action_dim = get_env_config(cube_size)
    hidden_dim = cfg['model']['hidden_dim']

    global_deepcube = DeepCube(state_dim, action_dim, hidden_dim).to(device).share_memory()
    global_epoch = mp.Value('i', 0)

    optimizer = optim_func(global_deepcube, learning_rate)
    lr_scheduler = scheduler_func(optimizer)

    BaseManager.register('defaultdict', defaultdict, DictProxy)
    mgr = BaseManager()
    mgr.start()
    loss_history = mgr.defaultdict(dict)
    valid_history = mgr.defaultdict(dict)

    os.makedirs(video_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(progress_path, exist_ok=True)

    if args.resume:
        checkpoint = torch.load(args.path)
        global_epoch = checkpoint['epoch']+1
        global_deepcube.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    workers = [Agent(global_deepcube, optimizer, lr_scheduler, global_epoch, valid_history, loss_history, cfg) 
                    for i in range(num_processes)]
    [w.start() for w in workers]
    [w.join() for w in workers]


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
    # TODO: 비디오 저장이 가능하도록
    solve_percentage_list = []
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
                # env.close_render()
                pass
        solve_percentage = (solve_count/sample_cube_count) * 100
        solve_percentage_list.append(solve_percentage)
    valid_history[epoch] = {'solve_percentage':solve_percentage_list}

class Agent(mp.Process):
    def __init__(self, global_deepcube, optimizer, lr_scheduler, global_epoch, valid_history, loss_history, cfg):
        super().__init__()
        self.device = torch.device('cpu')
        self.cfg = cfg
        self.batch_size = cfg['train']['batch_size']
        sample_size = cfg['train']['sample_size']
        self.epochs = cfg['train']['epochs']
        self.sample_epoch = cfg['train']['sample_epoch']
        self.sample_scramble_count = cfg['train']['sample_scramble_count']
        self.sample_cube_count = cfg['train']['sample_cube_count']
        buffer_size = cfg['train']['buffer_size']
        self.temperature = cfg['train']['temperature']
        self.validation_epoch = cfg['train']['validation_epoch']
        self.video_path = cfg['train']['video_path']
        self.model_path = cfg['train']['model_path']
        self.progress_path = cfg['train']['progress_path']
        cube_size = cfg['env']['cube_size']
        state_dim, action_dim = get_env_config(cube_size)
        hidden_dim = cfg['model']['hidden_dim']

        self.global_deepcube = global_deepcube
        self.deepcube = DeepCube(state_dim, action_dim, hidden_dim).to(self.device)
        self.deepcube.load_state_dict(self.global_deepcube.state_dict())
        self.env = make_env(self.device, cube_size)
        self.global_epoch = global_epoch
        self.local_epoch = 0

        self.optimizer = optimizer
        self.criterion_list = loss_func()
        self.lr_scheduler = lr_scheduler

        self.replay_buffer = ReplayBuffer(buffer_size, sample_size)
        self.valid_history = valid_history
        self.loss_history = loss_history

    def run(self):
        import time
        a = time.time()
        while self.global_epoch.value <= (self.epochs+1):
            with self.global_epoch.get_lock():
                self.global_epoch.value += 1
                self.local_epoch = self.global_epoch.value
            if (self.local_epoch-1) % self.sample_epoch == 0:
                self.env.get_random_samples(self.replay_buffer, self.deepcube, self.sample_scramble_count, self.sample_cube_count, self.temperature)
            loss = update_params(self.deepcube, self.global_deepcube, self.replay_buffer, self.criterion_list, self.optimizer, self.batch_size, self.device, self.temperature)
            self.loss_history[self.local_epoch] = {'loss':[loss]}
            if self.local_epoch % self.validation_epoch == 0:
                print(f'Current epochs: {self.local_epoch}, {time.time() - a}')
                validation(self.deepcube, self.env, self.valid_history, self.local_epoch, self.device, self.cfg)
                plot_valid_hist(self.valid_history, save_file_path=self.progress_path, validation_epoch=self.validation_epoch)
                save_model(self.deepcube, self.local_epoch, self.optimizer, self.lr_scheduler, self.model_path)
                plot_progress(self.loss_history, save_file_path=self.progress_path)
            self.lr_scheduler.step()
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default='', help='Path to pretrained model file')
    args = parser.parse_args()

    with open('./config/config.yaml') as f:
        cfg = yaml.safe_load(f)
    import time
    a = time.time()
    # train(cfg, args)
    # print(time.time()-a)
    multi_train(cfg, args)