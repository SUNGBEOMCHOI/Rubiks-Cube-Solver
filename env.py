import math
from collections import namedtuple

import numpy as np
import gym

from simulation.py222.py222 import initState, getOP, doMove, isSolved, getStickers, printCube
from simulation.gym_cube.gym_cube.envs.assets.cube_interactive import Cube as RenderCube
from utils import *
from model import DeepCube


def make_env(device, cube_size=3):
    """
    Make gym environment
    Args:
        env_name: Environment name you want to make
        cfg: config information which contains cube size
    Returns:
        env: gym environment
    """
    # TODO: gym make 할때 cube size를 넣어서 큐브를 생성
    # env = gym.make(env_name, cube_size)
    env = Cube(device, cube_size)
    return env

class Cube(gym.Env):
    def __init__(self, device, cube_size=3):
        """
        Gym environment for cube

        Args:
            cube_size: Cube size you want to make environment
        """
        super().__init__()
        self.transaction = namedtuple('Sample', ['state', 'target_value', 'target_policy', 'scramble_count', 'error'])
        self.cube_size = cube_size
        self.device = device
        self.action_to_sim_action = {\
            2:["U","U'","F","F'","R","R'"],
            3:["U","U'","F","F'","R","R'","D","D'","B","B'","L","L'"],\
            'render': [["U",1],["U",-1],["F",1],["F",-1],["R",1],["R",-1],["D",1],["D",-1],["B",1],["B",-1],["L",1],["L",-1]]
        }
        self.show_cube = False
        self.state_dim, self.action_dim = get_env_config(cube_size)
        self.init_state() # initialize cube, simulation cube, rendering cube

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
        action_sequence = np.random.randint(self.action_dim, size=scramble_count)
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
        info = {}
        if self.cube_size == 2:
            sim_action = self.action_to_sim_action[self.cube_size][action]
            self.sim_cube = doMove(self.sim_cube, sim_action)
            self.cube = self.sim_state_to_state(self.sim_cube)
            if isSolved(self.sim_cube):
                done = True
                reward = 1.0
            else:
                done = False
                reward = -1.0
        elif self.cube_size == 3:
            raise NotImplementedError
        else:
            raise NotImplementedError

        if self.show_cube:
            face, degree = self.action_to_sim_action['render'][action]
            self.fig.axes[3].rotate_face(face, degree, layer=0)
        return self.cube, reward, done, info

    def init_state(self):
        """
        Initialize state
        """
        if self.cube_size == 2:
            self.sim_cube = initState()
            self.cube = self.sim_state_to_state(self.sim_cube)
        elif self.cube_size == 3:
            raise NotImplementedError
        else:
            raise NotImplementedError

        if self.show_cube:
            self.render_cube = RenderCube(self.cube_size)
            self.fig = self.render_cube.draw_interactive()


    def render(self):
        """
        Render the environment to the screen
        Make matplot figure and show cube step
        """
        self.render_cube = RenderCube(self.cube_size)
        self.fig = self.render_cube.draw_interactive()
        self.show_cube=True

    def close_render(self):
        """
        Finish render mode
        """
        self.show_cube=False

    def save_video(self, video_path):
        """
        Save playing video to specific path

        Args:
            video_path: Path to save video
        """
        pass

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
                sample = self.transaction(state, target_value, target_policy, scramble_idx+1, error)
                replay_buffer.append(sample)
        
                

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
                state[state_cubelet][state_position] = 1.0

        elif self.cube_size == 3:
            raise NotImplementedError
        else:
            raise NotImplementedError
        return state

    def state_to_sim_state(self, state):
        """
        Return simulation state from our state

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
                raise NotImplementedError
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
        # weight = scramble_count ** (-1*temperature)
        with torch.no_grad():
            state_tensor = torch.tensor(self.cube, device=self.device).float()
            value, _ = model(state_tensor)
            error = abs(value.detach().item() - target_value)
        error = 0.0
        return target_value, target_policy, error

if __name__ == "__main__":
    import time
    deepcube = DeepCube([7, 21], 6, [256, 64, 32])
    cube = Cube('cpu', cube_size=2)
    replay_buffer = ReplayBuffer(500, 50)
    t = torch.randn((128, 7, 21))
    # a = time.time()
    # for _ in range(80):
    #     deepcube(t)
    # for _ in range(12000):
    #     cube.sim_state_to_state(cube.sim_cube)
    cube.get_random_samples(replay_buffer, deepcube, sample_scramble_count=20, sample_cube_count=100, temperature=0.3)
    # print(time.time() - a)
    # print(cube.sim_cube)
    # op = getOP(cube.sim_cube)
    # print(op)
    # print(getStickers(op))
    # print(cube.state_to_sim_state(cube.cube))
    