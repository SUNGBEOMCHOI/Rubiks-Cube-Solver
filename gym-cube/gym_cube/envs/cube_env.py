import gym
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from assets.cube_interactive import Cube as RenderCube
from assets.py222.py222 import initState, getOP, doMove, isSolved, getStickers, printCube
from utils import *
from assets.py333 import initState_3, doMove_3, getOP_3, isSolved_3, pos_to_state_3

class CubeEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array']}
    def __init__(self, device, cube_size=2):
        """
        Gym environment for cube

        Args:
            device: Torch device for training, eg.torch.device('cpu:0')
            cube_size: Cube size you want to make environment
        """
        self.cube_size = cube_size
        self.device = device
        self.action_to_sim_action = {\
            2:["U","U'","F","F'","R","R'"],
            3:["U","U'","F","F'","R","R'","D","D'","B","B'","L","L'"],\
            'render': [["U",1],["U",-1],["F",1],["F",-1],["R",1],["R",-1],["D",1],["D",-1],["B",1],["B",-1],["L",1],["L",-1]]
        }
        self.show_cube = False
        self.state_dim, self.action_dim = get_env_config(cube_size)
        self.init_state()
    
    def init_state(self):
        """
        Initialize state
        """
        if self.cube_size == 2:
            self.sim_cube = initState() # py222
            self.cube = self.sim_state_to_state(self.sim_cube)
        elif self.cube_size == 3:
            self.sim_cube = initState_3()
            self.cube = self.sim_state_to_state(self.sim_cube)
        else:
            raise NotImplementedError
        if self.show_cube:
            self.render_cube = RenderCube(self.cube_size)
            self.render_cube.env = self
            self.fig = self.render_cube.draw_interactive()        

    def reset(self, seed=None, scramble_count=2):
        """
        Reset the state to random scrambled cube

        Args:
            seed: Random seed number
            scramble_count: Number of scramble cubes randomly

        Return:
            Initial state shape [number of cublets, possible locations]
        """
        self.init_state()
        origin_state = np.random.get_state()
        if seed is not None:
            np.random.seed(seed)
        action_sequence = np.random.randint(self.action_dim, size = scramble_count)
        for action in action_sequence:
            state, _, _, _ = self.step(action)
        np.random.set_state(origin_state)
        return state
    
    def step(self, action):
        """
        Execute one time step within the environment

        Args:
            action: Integer of action you want to perform
            action can be 0 to 11 which relative to [U,U',F,F',R,R',D,D',B,B',L,L']
            
        Returns:
            state: Numpy array of state after action is performed
            reward: Return +1.0 if state is goal state, else return -1.0
            done: Return true if state is goal state, else return false
            info: Dictionary of useful information
        """
        if self.cube_size == 2:
            sim_action = self.action_to_sim_action[self.cube_size][action] # index에 대한 string 반환
            self.sim_cube = doMove(self.sim_cube, sim_action) # 각 string에 맞게 number가 배정되어 있고 이를 move definition 순서에 맞게 state를 변환
            self.cube = self.sim_state_to_state(self.sim_cube) # sticker index를 cube state로 변환
            if isSolved(self.sim_cube):
                done = True
                reward = 1.0
            else:
                done = False
                reward = -1.0
        elif self.cube_size == 3:
            sim_action = self.action_to_sim_action[self.cube_size][action]
            self.sim_cube = doMove_3(self.sim_cube, sim_action)
            self.cube = self.sim_state_to_state(self.sim_cube)
            if isSolved_3(self.sim_cube):
                done = True
                reward = 1.0
            else:
                done = False
                reward = -1.0
        else:
            raise NotImplementedError
        
        if self.show_cube:
            face, degree = self.action_to_sim_action['render'][action] # face string , degree number
            self.fig.axes[3].rotate_face(face, degree, layer = 0)
        return self.cube, reward, done, {}

    def render(self, mode=None):
        """
        Render the environment to the screen
        Make matplot figure and show cube step
        """
        self.render_cube = RenderCube(self.cube_size)
        self.fig = self.render_cube.draw_interactive()
        self.show_cube=True
        plt.close()
        return self.fig
    
    def close_render(self):
        """
        Finish render mode
        """
        self.show_cube = False
        plt.pause(1)
        plt.close()
    
    def sim_state_to_state(self, sim_state):
        """
        Return our state from simulation state

        Args:
            sim_state: Numpy array of simulation state
        
        Returns:
            Numpy array of our state
        """
        if self.cube_size == 2:
            state = np.zeros(self.state_dim)
            for position, cubelet in enumerate(getOP(sim_state)):
                state_cubelet, position_idx = cubelet
                state_position = position * 3 + position_idx
                state[state_cubelet][state_position] = 1.0 # one hot
        elif self.cube_size == 3:
            state = pos_to_state_3(getOP_3(sim_state))
        else:
            raise NotImplementedError
        return state
    
    def state_to_sim_state(self, state):
        """
        Return simulation state from our state

        Args:
            state: Our state you want to transform to simulation state

        Returns:
            Numpy array of simulation state
        """
        if self.cube_size == 2:
            sim_state = np.zeros((7, 2), dtype=np.int32)
            for cubelet, state_position in enumerate(state):
                state_position = np.where(state_position==1.0)[0][0]
                position, position_idx = state_position//3, state_position%3
                sim_state[position] = [cubelet, position_idx]
            sim_state = getStickers(sim_state)
        elif self.cube_size == 3:
            raise NotImplementedError
        else:
            raise NotImplementedError
        return sim_state
    
    def get_random_samples(self, replay_buffer, model, sample_scramble_count, sample_cube_count, temperature):
        """
        Add samples to replay buffer which contain (state, target value, target policy, scramble count, error)  for training
        
        Args:
            replay_buffer: Replay buffer to save samples
            model: Current deep cube model
            sample_scramble_count: Number of scramble cubes randomly
            sample_cube_count: Number of cube samples
        """
        for sample_cube_idx in range(1, sample_cube_count+1):
            self.init_state()
            action_sequence = np.random.randint(self.action_dim, size=sample_scramble_count)
            for scramble_idx, action in enumerate(action_sequence):
                state, _, _, _ = self.step(action)
                target_value, target_policy, error = self.get_target_value(model, scramble_idx+1, temperature)
                sample = {'state':state, 'target_value':target_value, 'target_policy':target_policy, 'scramble_count':scramble_idx+1, 'error':error}
                replay_buffer.append(sample)

    def get_target_value(self, model, scramble_count, temperature):
        """
        Return target value and target policy

        Args:
            state_list: List of states you want to get target valu and target policy
            model: Current deep cube model
            scramble_count: Scramble count of sample 

        Returns:
            target_value
            target_policy
            error: Difference between state value and target value
        """
        reward_list = []
        next_state_list = []
        for action in range(self.action_dim):
            if self.cube_size == 2:
                sim_action = self.action_to_sim_action[self.cube_size][action]
                next_sim_cube = doMove(self.sim_cube, sim_action)
                next_state = self.sim_state_to_state(next_sim_cube)
                if isSolved(next_sim_cube):
                    reward = 1.0
                    target_value, target_policy = 1.0, action
                    break
                else:
                    reward = -1.0
                next_state_list.append(next_state)
                reward_list.append(reward)
            elif self.cube_size == 3:
                sim_action = self.action_to_sim_action[self.cube_size][action]
                next_sim_cube = doMove_3(self.sim_cube, sim_action)
                next_state = self.sim_state_to_state(next_sim_cube)
                if isSolved_3(next_sim_cube):
                    reward = 1.0
                    target_value, target_policy = 1.0, action
                    break
                else:
                    reward = -1.0
                next_state_list.append(next_state)
                reward_list.append(reward)
            else:
                raise NotImplementedError
        if reward != 1.0:
            next_state_tensor = torch.tensor(np.array(next_state_list), device=self.device).float() # action_dim, state_dim[0], state_dim[1]
            reward_tensor = torch.tensor(reward_list, device = self.device) # action_dim
            with torch.no_grad():
                next_value, _ = model(next_state_tensor)
                value = next_value.squeeze(dim=-1).detach() + reward_tensor
            target_value, target_policy = torch.max(value, -1, keepdim=True)
            target_value, target_policy = target_value.item(), target_policy.item()
        weight = scramble_count ** (-1*temperature)
        with torch.no_grad():
            state_tensor = torch.tensor(self.cube, device=self.device).float()
            value, _ = model(state_tensor)
            error = abs(value.detach().item() - target_value) * weight
        return target_value, target_policy, error
    
    def save_video(self, cube_size, scramble_count, sample_cube_count, video_path='./video'):
        """
        Save inference video of trained model

        Args:
            cube_size
            scramble_count: Scramble count of cube
            sample_cube_count: Index of cube
            video_path: Path for saving video
        """
        filename = f'cube{cube_size}_scramble{scramble_count}_sample{sample_cube_count}.gif'
        frames = self.fig.axes[3].frames # interactivecube.frames
        plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
        patch = plt.imshow(frames[0])
        plt.axis('off')

        def animate(i):
            patch.set_data(frames[i])

        anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=1)
        anim.save(video_path + '/'+ filename, writer='imagemagick', fps=60)
        self.close_render()