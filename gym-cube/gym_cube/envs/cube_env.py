from unicodedata import name
import gym
import numpy as np
import matplotlib.pyplot as plt
import os
from gym_cube.envs.assets.cube_interactive import InteractiveCube, Cube

from collections import namedtuple
from gym.wrappers.monitoring.video_recorder import VideoRecorder

'''
UDFBRLU'D'F'B'R'L'
01234567891011
action = {'U':0}


'''
class CubeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, cube_size = 3):
        self.cube = Cube(cube_size)
        self.icube = InteractiveCube(cube=self.cube)
        self.N = cube_size
        self.state = [self.cube._face_centroids, self.cube._faces, self.cube._sticker_centroids, self.cube._stickers]
        self.num_turns = 0

        self.transaction = namedtuple('Point', ['state', 'target_value', 'target_policy', 'scramble_count'])


    def step(self, action):
        face, layer, degree = action
        # self.cube.move(face, layer, degree)
        self.cube.rotate_face(face, degree, layer)
        self.num_turns += 1
        # check done
        done = 1
        reward = 0
        # for i in range(len(self.state)):
        #     side = self.state[i]
        #     if side[0][0] == (np.sum(side) / (self.N * self.N)): 
        #         done *= 1
        #     else:
        #         done *= 0
        # # reward shaping
        # if done == 1:
        #     reward += 1
        # else:
        #     reward -= 1
        return self.state, reward, done
    
    def reset(self):
        # self.state = np.array([np.tile(i, (self.N, self.N)) for i in range(6)])
        self.cube._initialize_arrays()
        self.num_turns = 0
        return self.state, self.num_turns

    
    def randomize(self, seed = None ,scramble_count=1000):
        # for t in range(scramble_count):
        #     face = self.cube.dictface[np.random.randint(6)]
        #     layer = np.random.randint(self.N)
        #     degree = 1 if np.random.random() < 0.5 else -1
        #     self.cube.move(face, layer, degree)
        # self.render_mode((0, 1))

        self.icube._random_view(steps = scramble_count)

    def render_mode(self, mode):
        # flat = mode[0]
        # views = mode[1]
        # self.cube.render(flat, views)
        # self.icube._draw_cube()
        # plt.show()
        pass
    
    def render(self):
        fig = self.cube.draw_interactive()
        plt.show()

    def save_video(self, env, policy, video_path):
        recorder = VideoRecorder(env=env, path=video_path, metadata={'num_turns': self.num_turns})
        state = env.reset()
        num_actions = len(policy)
        i = 0
        while i < num_actions:
            print(i)
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
