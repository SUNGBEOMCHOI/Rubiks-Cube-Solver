from collections import deque

import torch
from torch.utils.data import Dataset, DataLoader

def loss_func():
    """
    Return value, policy criterion
    Returns:
        loss_list: List contains value_criterion and policy_criterion
    """
    pass

def optim_func(model, learning_rate):
    """
    Return value, policy optimizer
    Returns:
        optim_list: List contains value_optim, policy_optim
    """
    # TODO MSE loss, crossentropy의 reduction은 none으로 설정(weighted loss 계산을 위함)
    pass

def scheduler_func(optim_list):
    """
    Return value, policy learning rate scheduler
    Args:
        optim_list: List of value, policy optimizer
    Returns:
        scheduler_list: List contains value, policy learning rate scheduler
    """
    pass

def plot_progress(loss_history, save_file_path='./train_progress'):
    """
    plot train progress, x-axis: epoch, y-axis: loss
    Example of loss_history: {1:{loss:[5.2]}, 2:{loss:[3.1]}, 3:{loss:[1.5]}, ...}
    
    Args:
        loss_history: Dictionary which contains loss
        save_file_path: Path for saving progress graph
    """
    pass

def plot_valid_hist(valid_history, save_file_path='./train_progress'):
    """
    plot validation results, x-axis: scramble distance, y-axis: percentage solved
    Example of valid_history: {1:{solve_percentage:[10, 8, 5, ...]}, 2:{solve_percentage:[30, 24, 10, ...]}, 3:{solve_percentage:[50, 35, 22, ...]}, ...}
    
    Args:
        valid_history: Dictionary which contains solved percentage for each scramble distance
        save_file_path: Path for saving progress graph
    """
    pass

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
        -> ([7, 21], 6)
    """
    pass

def save_model(model, epoch, optim_list, lr_scheduler_list, model_path):
    """
    Save trained model

    Args:
        model: Model you want to save
        epoch: Current epoch
        optim_list: List contains value_optim, policy_optim
        lr_scheduler_list: List contains value, policy learning rate scheduler
        model_path: Path to save the model
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict' : model.state_dict(),
        'value_optimizer_state_dict': optim_list[0].state_dict(),
        'policy_optimizer_state_dict': optim_list[1].state_dict(),
        'value_lr_scheduler': lr_scheduler_list[0].state_dict(),
        'policy_lr_scheduler': lr_scheduler_list[1].state_dict()
        }, f'{model_path}/checkpoint_{epoch}.pt')

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
        pass

    def append(self, x):
        """
        Append x into replay memory
        Args:
            x: Input
        """
        self.memory.append(x)
        
def update_params(model, replay_buffer, criterion_list, optim_list, batch_size, device):
    """
    Update model networks' parameters with replay buffer
    
    Args:
        model: DeepCube model
        replay_buffer: Replay memory that contains date samples
        criterion_list: List contains value_criterion, policy_criterion
        optim_list : criterion_list: List contains value_optimizer, policy_optimizer
        batch_size
        device

    Returns:
        total_loss: sum of value loss and policy loss
    """
    value_criterion, policy_criterion = criterion_list()
    value_optim, policy_optim = optim_list()

    train_dataloader = DataLoader(replay_buffer, batch_size=batch_size, shuffle=True)
    num_samples = len(replay_buffer)
    total_loss = 0.0
    for state, target_value, target_policy, scramble_count in train_dataloader:
        state = state.to(device)
        target_value = target_value.to(device)
        target_policy = target_policy.to(device)
        scramble_count = scramble_count.to(device) # [B]
        reciprocal_scramble_count = torch.reciprocal(scramble_count)

        # update value network
        predicted_value, predicted_policy = model(state.detach())
        value_optim.zero_grad()
        value_loss = (value_criterion(predicted_value, target_value.detach()).squeeze(dim=-1) * \
                        reciprocal_scramble_count.squeeze(dim=-1).detach()).mean()
        value_loss.backward(retain_graph=True)
        value_optim.step()

        # update policy network
        policy_optim.zero_grad()
        policy_loss = (policy_criterion(predicted_policy, target_policy.detach()) * \
                        reciprocal_scramble_count.squeeze(dim=-1).detach()).mean()
        policy_loss.backward(retain_graph=True)
        policy_optim.step()

        total_loss = total_loss + value_loss.item() + policy_loss.item()
    total_loss/= num_samples
    return total_loss