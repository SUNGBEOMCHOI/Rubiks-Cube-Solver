from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

def loss_func():
    """
    Return value, policy criterion type of torch.nn loss function
    Returns:
        loss_list: List contains value_criterion and policy_criterion
    """
    value_criterion = nn.MSELoss(reduction='none')
    policy_criterion = nn.CrossEntropyLoss(reduction='none')
    return [value_criterion, policy_criterion]

def optim_func(model, learning_rate):
    """
    Return optimizer
    Returns:
        optimizer
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return optimizer

def scheduler_func(optimizer):
    """
    Return value, policy learning rate scheduler
    Args:
        optimizer
    Returns:
        scheduler: learning rate scheduler
    """
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.001, step_size_up=50, step_size_down=100, mode='triangular')
    return scheduler

def plot_progress(loss_history, save_file_path='./train_progress'):
    """
    plot train progress, x-axis: epoch, y-axis: loss
    
    Args:
        loss_history: Dictionary which contains loss
        save_file_path: Path for saving progress graph
    """
    epoch_list = []
    loss_list = []
    for epoch, value in loss_history.items():
        epoch_list.append(epoch)
        loss_list.append(value['loss'])
    epoch_list = np.array(epoch_list)
    loss_list = np.array(loss_list)
    plt.plot(epoch_list, loss_list)
    plt.title('Train Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(f'{save_file_path}/{epoch}_train.png')
    plt.show()

def plot_valid_hist(valid_history, save_file_path='./train_progress'):
    """
    plot validation results, x-axis: scramble distance, y-axis: percentage solved
    
    Args:
        valid_history: Dictionary which contains solved percentage for each scramble distance
        save_file_path: Path for saving progress graph
    """
    for epoch, solve_percentage in valid_history.items():
        scramble_count_list = np.array(1, len(solve_percentage)+1, dtype=np.int32)
        solve_percentage_list = np.array(solve_percentage['solve_percentage'])
        plt.plot(scramble_count_list, solve_percentage_list, label=str(epoch))
    plt.title('Solve percentage')
    plt.xlabel('Scramble count')
    plt.ylabel('Solve percentage')
    plt.legend()
    plt.savefig(f'{save_file_path}/{epoch}_solve_percentage.png')
    plt.show()

def get_env_config(cube_size=3):
    """
    Return state dimension and action dimension corresponding cube size

    Args:
        cube_size:
    
    Returns:
        state_dim: List of [number of cublets, possible locations]
        action_dim: Number of actions you can take

    Examples:
        get_env_config(cube_size=2)
        -> ([7, 21], 6)
        get_env_config(cube_size=2)
        -> ([20, 24], 12)
    """
    if cube_size == 2:
        state_dim = [7, 21]
        action_dim = 6
    elif cube_size == 3:
        state_dim = [20, 24]
        action_dim = 12
    else:
        assert AssertionError
    
    return state_dim, action_dim

def save_model(model, epoch, optimizer, lr_scheduler, model_path='./pretrained'):
    """
    Save trained model

    Args:
        model: Model you want to save
        epoch: Current epoch
        optimizer
        lr_scheduler
        model_path: Path to save model
    """
    torch.save({
        'epoch' : epoch,
        'model_state_dict' : model.state_dict(),
        'optimizer_state_dict' : optimizer.state_dict(),
        'lr_scheduler' : lr_scheduler.state_dict(),
        }, f'{model_path}/model_{epoch}.pt')

class ReplayBuffer(Dataset):
    def __init__(self, buf_size):
        """
        Make replay buffer of deque structure which stores data samples
        
        Args:
            buf_size: deque max size
        """
        self.memory = deque(maxlen=buf_size)

    def __len__(self):
        """
        Return length of replay memory

        Returns:
            Length of replay memory        
        """
        return len(self.memory)

    def __getitem__(self, idx):
        """
        Get samples the sample corresponding to the index

        Args:
            idx: index of sample you want to get

        Returns:
            state_tensor: Torch tensor of state of shape [state_dim]
            target_value_tensor: Torch tensor of target value of shape [1]
            target_policy_tensor: Torch tensor of target value of shape [action_dim]
            scramble_count_tensor: Torch tensor of scamble count of shape [1]
        """
        # TODO shape 확인해봐야함
        state_tensor = torch.tensor(self.memory[idx].state)
        target_value_tensor = torch.tensor(self.memory[idx].target_value)
        target_policy_tensor = torch.tensor(self.memory[idx].target_policy)
        scramble_count_tensor = torch.tensor(self.memory[idx].scramble_count)
        return state_tensor, target_value_tensor, target_policy_tensor, scramble_count_tensor
        

    def append(self, x):
        """
        Append x into replay memory
        Args:
            x: Input
        """
        self.memory.append(x)
        
def update_params(model, replay_buffer, criterion_list, optimizer, batch_size, device):
    """
    Update model networks' parameters with replay buffer
    
    Args:
        model: DeepCube model
        replay_buffer: Replay memory that contains date samples
        criterion_list: List contains value_criterion, policy_criterion
        optimizer
        batch_size
        device

    Returns:
        total_loss: sum of value loss and policy loss
    """
    value_criterion, policy_criterion = criterion_list()

    train_dataloader = DataLoader(replay_buffer, batch_size=batch_size, shuffle=True)
    num_samples = len(replay_buffer)
    total_loss = 0.0
    for state, target_value, target_policy, scramble_count in train_dataloader:
        state = state.to(device)
        target_value = target_value.to(device)
        target_policy = target_policy.to(device)
        scramble_count = scramble_count.to(device)
        reciprocal_scramble_count = torch.reciprocal(scramble_count)

        predicted_value, predicted_policy = model(state.detach())
        optimizer.zero_grad()
        # calculate value loss
        value_loss = (value_criterion(predicted_value, target_value.detach()).squeeze(dim=-1) * \
                        reciprocal_scramble_count.squeeze(dim=-1).detach()).mean()

        # calculate policy loss
        policy_loss = (policy_criterion(predicted_policy, target_policy.detach()) * \
                        reciprocal_scramble_count.squeeze(dim=-1).detach()).mean()
        loss = value_loss + policy_loss
        loss.backward()
        optimizer.step()

        total_loss = total_loss + loss.item()
    total_loss/= num_samples
    return total_loss