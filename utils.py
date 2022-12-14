import math
from collections import deque, Counter

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

class SharedAdam(optim.Adam):
    """
    Implements Adam algorithm with shared states.

    Args:
        params: Torch model parameters you want to train
        lr: learning rate
        betas: Coefficients used for computing running averages of gradient and its square
        eps: Term added to the denominator to improve numerical stability
        weight_decay: Weight decay (L2 penalty)
    """

    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr, betas, eps, weight_decay)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_()

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step'].item()
                bias_correction2 = 1 - beta2 ** state['step'].item()
                step_size = group['lr'] * math.sqrt(
                    bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss

def optim_func(model, learning_rate):
    """
    Return optimizer
    Returns:
        optimizer
    """
    optimizer = SharedAdam(model.parameters(), lr=learning_rate)
    return optimizer

def scheduler_func(optimizer):
    """
    Return value, policy learning rate scheduler
    Args:
        optimizer
    Returns:
        scheduler: learning rate scheduler
    """
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.005, step_size_up=200, step_size_down=400, mode='triangular', cycle_momentum=False)
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
    # plt.show()
    plt.close()

def plot_valid_hist(valid_history, save_file_path='./train_progress', validation_epoch=10):
    """
    plot validation results, x-axis: scramble distance, y-axis: percentage solved
    
    Args:
        valid_history: Dictionary which contains solved percentage for each scramble distance
        save_file_path: Path for saving progress graph
        validation_epoch
    """
    max_scramble_count = len(valid_history[validation_epoch]['solve_percentage'])
    plot_epoch_list = np.unique(np.linspace(1, len(valid_history)+0.001, num=5, dtype=int))*validation_epoch
    scramble_count_list = np.arange(1, max_scramble_count+1)
    for epoch in plot_epoch_list:
        solve_percentage_list = np.array(valid_history[epoch]['solve_percentage'])
        plt.plot(scramble_count_list, solve_percentage_list, label=str(epoch))
    plt.title('Solve percentage')
    plt.xlabel('Scramble count')
    plt.ylabel('Solve percentage')
    plt.legend()
    plt.savefig(f'{save_file_path}/{epoch}_solve_percentage.png')
    # plt.show()
    plt.close()

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

def save_model(model, epoch, optimizer, model_path):
    """
    Save trained model
    Args:
        model: Model you want to save
        epoch: Current epoch
        optimizer: Torch optimizer
        model_path: Path to save model
    """
    torch.save({
        'epoch' : epoch,
        'model_state_dict' : model.state_dict(),
        'optimizer_state_dict' : optimizer.state_dict(),
        }, f'{model_path}/model_{epoch}.pt')

class ReplayBuffer(Dataset):
    def __init__(self, buf_size, sample_size):
        """
        Make replay buffer of deque structure which stores data samples
        
        Args:
            buf_size: deque max size
        """
        self.buf_size = buf_size
        self.sample_size = sample_size
        self.memory = deque(maxlen=self.buf_size)
        self.error_memory = deque(maxlen=self.buf_size) # Deque of saving error
        # error means difference between target value and predicted value
        self.prioritized_idx = None

    def __len__(self):
        """
        Return length of prioritized memory
        Returns:
            Length of prioritized memory
        """
        return len(self.prioritized_idx)

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
        memory_idx = self.prioritized_idx[idx]
        state_tensor = torch.tensor(self.memory[memory_idx]['state'])
        target_value_tensor = torch.tensor(self.memory[memory_idx]['target_value'])
        target_policy_tensor = torch.tensor(self.memory[memory_idx]['target_policy'])
        scramble_count_tensor = torch.tensor(self.memory[memory_idx]['scramble_count'])
        idx_tensor = torch.tensor(memory_idx)
        return state_tensor, target_value_tensor, target_policy_tensor, scramble_count_tensor, idx_tensor
        
    def get_prioritized_sample(self):
        if len(self.memory) <= self.sample_size:
            self.prioritized_idx = np.arange(len(self.memory))
        else:
            np_error_memory = np.array(self.error_memory)
            self.prob_memory = np_error_memory / sum(np_error_memory)
            self.prioritized_idx = np.random.choice(np.arange(len(self.memory)), self.sample_size, replace=False, p=self.prob_memory)

    def append(self, x):
        """
        Append x into replay memory and save error list
        Args:
            x: Input
        """
        self.memory.append(x)
        self.error_memory.append(x['error'])

    def update(self, idx, error):
        """
        Update error of replay buffer

        Args:
            idx: Index of replay memory you want to change
            error: New error
        """
        self.error_memory[idx] = error

        
def update_params(model, replay_buffer, criterion_list, optimizer, batch_size, device, temperature, global_model=None):
    """
    Update model networks' parameters with replay buffer
    
    Args:
        model: DeepCube model
        replay_buffer: Replay memory that contains date samples
        criterion_list: List contains value_criterion, policy_criterion
        optimizer
        batch_size
        device
        temperature: Constant of scramble count weight
                     0 -> all scramble count has same weight, Large temperature has large difference weight
        global_model: Gloabal model for using multi processing
        
    Returns:
        total_loss: sum of value loss and policy loss
    """
    value_criterion, policy_criterion = criterion_list

    replay_buffer.get_prioritized_sample()
    train_dataloader = DataLoader(replay_buffer, batch_size=batch_size, shuffle=True)
    num_samples = len(replay_buffer)
    total_loss = 0.0
    for state, target_value, target_policy, scramble_count, memory_idxs in train_dataloader:
        state = state.to(device)
        target_value = target_value.to(device)
        target_policy = target_policy.to(device)
        scramble_count = scramble_count.to(device)
        reciprocal_scramble_count = torch.pow(scramble_count, -temperature)
        
        predicted_value, predicted_policy = model(state.float().detach())
        predicted_value, predicted_policy = predicted_value.squeeze(dim=-1), predicted_policy.squeeze(dim=-1)
        # calculate value loss
        loss = value_criterion(predicted_value, target_value.detach()).squeeze(dim=-1) *\
                                reciprocal_scramble_count.squeeze(dim=-1).detach()
                        
        for loss_idx, memory_idx in enumerate(memory_idxs):
            replay_buffer.update(memory_idx, loss[loss_idx].item())
        value_loss = loss.mean()

        # calculate policy loss
        policy_loss = (policy_criterion(predicted_policy, target_policy.detach()) * \
                        reciprocal_scramble_count.squeeze(dim=-1).detach()).mean()
        loss = value_loss + policy_loss
        
        if global_model:
            optimizer.zero_grad()
            loss.backward()
            for local_param, global_param in zip(model.parameters(), global_model.parameters()):
                global_param._grad = local_param.grad
            optimizer.step()
            model.load_state_dict(global_model.state_dict())
        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss = total_loss + loss.item()
    total_loss/= num_samples
    return total_loss