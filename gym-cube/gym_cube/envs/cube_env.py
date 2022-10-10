from unicodedata import name
import gym
import numpy as np
import matplotlib.pyplot as plt
import os
from gym_cube.envs.assets.cube import Cube 

from collections import namedtuple
from gym.wrappers.monitoring.video_recorder import VideoRecorder



class CubeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, cube_size = 3):
        self.cube = Cube(cube_size)
        self.N = cube_size
        self.state = self.cube.stickers
        self.done = 1
        self.reward = 0
        self.num_turns = 0

        self.transaction = namedtuple('Point', ['state', 'target_value', 'target_policy', 'scramble_count'])


    def step(self, action):
        face, layer, degree = action
        self.cube.move(face, layer, degree)
        self.num_turns += 1
        # check done
        for i in range(len(self.state)):
            side = self.state[i]
            if side[0][0] == (np.sum(side) / (self.N * self.N)): 
                self.done *= 1
            else:
                self.done *= 0
        # reward shaping
        if self.done == 1:
            self.reward += 1
        else:
            self.reward -= 1
        return self.state, self.reward, self.done
    
    def reset(self):
        self.state = np.array([np.tile(i, (self.N, self.N)) for i in range(6)])
        self.done = 0
        self.reward = 0
        self.num_turns = 0
        return self.state
    
    def randomize(self, seed = None ,scramble_count=1000):
        for t in range(scramble_count):
            face = self.cube.dictface[np.random.randint(6)]
            layer = np.random.randint(self.N)
            degree = 1 if np.random.random() < 0.5 else -1
            self.cube.move(face, layer, degree)
        self.render((0, 1))

    def render_mode(self, mode):
        flat = mode[0]
        views = mode[1]
        self.cube.render(flat, views)
        plt.show()
    
    def render(self):
        flat = False
        views = True
        self.cube.render(flat, views)
        plt.show()

    def save_video(self, env, policy, video_path):
        recorder = VideoRecorder(env=env, path=video_path, metadata={'num_turns': self.num_turns})
        state = env.reset()
        num_actions = len(policy)
        i = 0
        while i < num_actions:
            recorder.capture_frame()
            action = policy[i]
            state, reward, done = env.step(action)
            i += 1
            if done:
                state = env.reset()
                print("Saved Video")
                recorder.close()
                recorder.enabled = False


        

    def get_random_samples(self, replay_buffer, scramble_count, sample_cube_count):
        pass
