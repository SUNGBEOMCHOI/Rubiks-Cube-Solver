import torch
import torch.nn as nn

class DeepCube(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.encoder_net = nn.Sequential()
        self.policy_net = nn.Sequential()
        self.value_net = nn.Sequential()

    def forward(self, x):
        """
        Return action probability and state values corresponding input states

        Args:
            x: input state of size [batch_size, state_dim[0], state_dim[1]]
        Returns:
            value: Torch tensor of state value of size [batch_size, 1]
            action_probs: Torch tensor of action probability of size [batch_size, action_dim]
        """
        x = self.encoder_net(x)
        value = self.value_net(x)
        action_probs = self.policy_net(x)
        return value, action_probs

    def get_action(self, x):
        """
        Return action corresponding input states

        Args:
            x: input state of size [state_dim[0], state_dim[1]]
        Returns:
            action: Integer of action
        """
        pass