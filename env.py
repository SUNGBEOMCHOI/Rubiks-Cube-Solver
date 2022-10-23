from collections import namedtuple

import numpy as np
import gym

from simulation.py222.py222 import initState, getOP, doMove, isSolved, new_normFC, printCube
from simulation.gym_cube.gym_cube.envs.assets.cube_interactive import Cube as RenderCube
from utils import *


def make_env(env_name, cube_size=3):
    """
    Make gym environment
    Args:
        env_name: Environment name you want to make
        cfg: config information which contains cube size
    Returns:
        env: gym environment
    """
    # TODO: gym make 할때 cube size를 넣어서 큐브를 생성
    env = gym.make(env_name, cube_size)
    return env

class Cube(gym.Env):
    def __init__(self, cube_size=3):
        """
        Gym environment for cube

        Args:
            cube_size: Cube size you want to make environment
        """
        super().__init__()
        self.transaction = namedtuple('Point', ['state', 'target_value', 'target_policy', 'scramble_count'])
        self.cube_size = cube_size
        self.action_to_sim_action = {\
            2:["U","U'","F","F'","R","R'"],
            3:["U","U'","F","F'","R","R'","D","D'","B","B'","L","L'"],\
            'render': [["U",1],["U",-1],["F",1],["F",-1],["R",1],["R",-1],["D",1],["D",-1],["B",1],["B",-1],["L",1],["L",-1]]
        }
        self.show_cube = False
        # self.state_dim, self.action_dim = get_env_config(cube_size=3)
        self.state_dim, self.action_dim = [7, 3], 6
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
            action can be 0 to 11 which relative to [U,D,F,B,R,L,U',D',F',B',R',L']
            
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
            self.sim_cube = new_normFC(self.sim_cube, sim_action)
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
        Make matplot figure and self.show_video=True

        """
        self.render_cube = RenderCube(self.cube_size)
        self.fig = self.render_cube.draw_interactive()
        self.show_cube=True

    def save_video(self, video_path):
        """
        Save playing video to specific path

        Args:
            video_path: Path to save video
        """
        pass

    def get_random_samples(self, replay_buffer, model, sample_scramble_count, sample_cube_count):
        """
        Add samples to replay buffer which contain state, target value, target policy for training
        
        Args:
            replay_buffer: Replay buffer to save samples
            model: Current deep cube model
            sample_scramble_count: Number of scramble cubes randomly
            sample_cube_count: Number of cube samples

        """
        pass
                

    def sim_state_to_state(self, sim_state):
        """
        Return our state from simulation state

        Args:
            sim_state: Numpy array of simulation state
        
        Returns:
            Numpy array of our state
        """
        if self.cube_size == 2:
            state = np.array(getOP(sim_state))
        elif self.cube_size == 3:
            raise NotImplementedError
        else:
            raise NotImplementedError
        return state

    def state_to_sim_state(self):
        """
        Return simulation state from our state

        Returns:
            Numpy array of simulation state
        """
        if self.cube_size == 2:
            # sim_state = self.cube
            raise NotImplementedError
        elif self.cube_size == 3:
            raise NotImplementedError
        else:
            raise NotImplementedError
        return sim_state

if __name__ == "__main__":
    cube = Cube(cube_size=2)
    cube.render()
    state = cube.reset()
    print(state)
    printCube(cube.sim_cube)