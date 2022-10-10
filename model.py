import torch
import torch.nn as nn

class DeepCube(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DeepCube, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.encoder_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.state_dim[0]*self.state_dim[1],  4096),
            nn.ELU(),
            nn.Linear(4096, 2048),
            nn.ELU()
        )
        self.policy_net = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ELU(),
            nn.Linear(512, self.action_dim)
        )
        self.value_net = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ELU(),
            nn.Linear(512, 1)
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
        x = self.encoder_net(x)
        value = self.value_net(x)
        action_output = self.policy_net(x)
        return value, action_output

    def get_action(self, x):
        """
        Return action corresponding input states

        Args:
            x: input state of size [state_dim[0], state_dim[1]]
        Returns:
            action: Integer of action
        """
        if x.dim() == 2: # batch size가 없으면
            x = x.unsqueeze(dim=0)
        _, action_output = self.forward(x)
        return torch.argmax(action_output).item()
