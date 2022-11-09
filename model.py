import copy

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

class DeepCube(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(DeepCube, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.encoder_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.state_dim[0]*self.state_dim[1],  self.hidden_dim[0]),
            nn.ELU(),
            nn.Linear(self.hidden_dim[0], self.hidden_dim[1]),
            nn.ELU()
        )
        self.policy_net = nn.Sequential(
            nn.Linear(self.hidden_dim[1], self.hidden_dim[2]),
            nn.ELU(),
            nn.Linear(self.hidden_dim[2], self.action_dim)
        )
        self.value_net = nn.Sequential(
            nn.Linear(self.hidden_dim[1], self.hidden_dim[2]),
            nn.ELU(),
            nn.Linear(self.hidden_dim[2], 1)
        )

    def forward(self, x):
        """
        Return action output and state values corresponding input states
        Args:
            x: input state of size [batch_size, state_dim[0], state_dim[1]]
        Returns:
            value: Torch tensor of state value of size [batch_size, 1]
            action_output: Torch tensor of action probability of size [batch_size, action_dim]
        """
        if x.dim() == 2: # batch size가 없으면
            x = x.unsqueeze(dim=0)
        x = self.encoder_net(x)
        value = self.value_net(x)
        action_output = self.policy_net(x)
        return value, action_output

    def get_action(self, x, pre_action=None):
        """
        Return action corresponding input states
        Args:
            x: Torch tensor of size [state_dim[0], state_dim[1]]
            pre_action(int): last action to avoid its counter action (만약 pre_action이 None이면 기존과 동일, 안 넣어도 됨)
        Returns:
            action: Integer of action
        """
        if x.dim() == 2: # batch size가 없으면
            x = x.unsqueeze(dim=0)
        _, action_output = self.forward(x)

        # action_probs = torch.nn.functional.softmax(action_output)
        # m = Categorical(action_probs)
        # action = m.sample().item()
        
        if pre_action == None :
            invalid_action = None
        elif pre_action % 2 == 1: # pre_action이 0이면 1 // pre_action이 1이면 0
            invalid_action = pre_action-1
        else:
            invalid_action = pre_action+1

        if invalid_action == action_output.sort(descending=True)[1][0][0]:
            action = action_output.sort(descending=True)[1][0][1].item() # 최선의 action이 counter일 때 차선의 action
        else:
            action = action_output.sort(descending=True)[1][0][0].item() # 최선의 action
 
        return action

    def predict(self, x):
        """
        Return action and value corresponding input states
        Args:
            x: input state of size [state_dim[0], state_dim[1]], tensor
        Returns:
            value : integer of value  size:(action_dim,)
            action : policy vector    size:(1,)
        """
        x = torch.tensor(x).float().detach()
        value, policy = self.forward(x)
        policy = nn.functional.softmax(policy, dim=-1)

        return value.numpy()[0], policy.numpy()[0]

    def get_action_with_2step(self, x, env):
        """
        Return action corresponding input states by 2 step state prediction

        Args:
            x: Torch tensor of size [state_dim[0], state_dim[1]]
            env: Deepcube environment
        Returns:
            action: Integer of action
        """
        state_list = []
        for action1 in range(self.action_dim):
            x = x.numpy()
            next_state, done = env.get_next_state(x, action1)
            state_list.append(next_state)
            if done:
                break
            for action2 in range(self.action_dim):
                next_next_state, done = env.get_next_state(next_state, action2)
                state_list.append(next_next_state)
                if done:
                    break
            if done:
                break
        if done:
            action = action1
        else:
            state_tensor = torch.tensor(np.array(state_list)).detach().float()
            with torch.no_grad():
                value, _ = self.forward(state_tensor)
            action = torch.argmax(value.squeeze(dim=1)).item() // 7
        return action