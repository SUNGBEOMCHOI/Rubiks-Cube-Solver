from tkinter.messagebox import NO
from typing import Optional
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
'''
class CubeEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array',]}

    def __init__(self, cube_size = 3, render_mode:Optional[str] = None):
        self.cube = Cube(cube_size)
        self.icube = InteractiveCube(cube=self.cube)
        self.N = cube_size
        self.state = [self.cube._face_centroids, self.cube._faces, self.cube._sticker_centroids, self.cube._stickers]
        self.num_turns = 0

        self.transaction = namedtuple('Point', ['state', 'target_value', 'target_policy', 'scramble_count'])

        self.render_mode = render_mode
        self.initial_cube = Cube(cube_size)
        self.iicube = InteractiveCube(cube = self.initial_cube)

    def step(self, action_key):
        action_dict = {0: "U", 1: "D", 2: "F", 3: "B", 4: "R", 5: "L", 6: "U'", 7: "D'", 8: "F'", 9:"B'", 10:"R'", 11: "L'"}
        action = action_dict[action_key]
        face_dict = {0: "U", 1: "D", 2: "F", 3: "B", 4: "R", 5: "L", 6: "U", 7: "D", 8: "F", 9:"B", 10:"R", 11: "L"}
        face = face_dict[action_key]
        # degree_dict = {"U": 1, "D": 1, "F": 1, "B": 1, "R": 1, "L": 1, "U'":-1, "D'": -1, "F'": -1, "B'": -1, "R'": -1, "L'": -1}
        degree_dict = {"U": -1, "D": -1, "F": -1, "B": -1, "R": -1, "L": -1, "U'":1, "D'": 1, "F'": 1, "B'": 1, "R'": 1, "L'": 1}
        degree = degree_dict[action]
        layer = 0
        self.cube.rotate_face(face, degree, layer)
        self.num_turns += 1
        # check done
        done = 1
        reward = 0
        for i in range(6): # 54
            # side = self.icube._project(self.state[3])[:,:,:2][9*(i): 9*(i+1)]
            side = self.icube._project(self.cube._stickers)[:,:,:2][9*(i): 9*(i+1)]
            initial = self.iicube._project(self.initial_cube._stickers)[:,:,:2][9*(i): 9*(i+1)]
            for _ in range(9):
                done *= np.array_equal(side[_], initial[_])
                

            # if side[0][0] == (np.sum(side) / (self.N * self.N)): 
            #     done *= 1
            # else:
            #     done *= 0
        # reward shaping
        if done == 1:
            reward = 1
        else:
            reward = -1
        info = None
        return [self.cube._face_centroids, self.cube._faces, self.cube._sticker_centroids, self.cube._stickers], done, reward, info
    
    def reset(self, seed = None, scramble_count=1000):
        face_list = ["U", "D", "L", "R", "B", "F"]
        for count in range(scramble_count):
            face = face_list[np.random.randint(6)]
            # layer = np.random.randint(3)
            degree = 1 if np.random.random() < 0.5 else -1
            self.cube.rotate_face(face, degree, 0)
        return
    
    def initialize(self):
        self.cube._initialize_arrays()
        self.num_turns = 0

    
    def randomize(self, seed = None ,scramble_count=1000):
        face_list = ["U", "D", "L", "R", "B", "F"]
        for count in range(scramble_count):
            face = face_list[np.random.randint(6)]
            # layer = np.random.randint(3)
            degree = 1 if np.random.random() < 0.5 else -1
            self.cube.rotate_face(face, degree, 0)

    def render_mode(self):
        # flat = mode[0]
        # views = mode[1]
        # self.cube.render(flat, views)
        # self.icube._draw_cube()
        # plt.show()
        pass
    
    def render(self):
        fig = self.cube.draw_interactive()
        plt.show()
        fig.canvas.draw()

        if self.render_mode is None:
            return
        else:
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1]+(3,))
            return data


    # def save_video(self, env, policy, video_path):
    #     recorder = VideoRecorder(env=env, path=video_path, metadata={'num_turns': self.num_turns})
    #     state = env.reset()
    #     num_actions = len(policy)
    #     i = 0
    #     while i < num_actions:
    #         print(i)
    #         recorder.capture_frame()
    #         action = policy[i]
    #         state, reward, done = env.step(action)
    #         i += 1
    #         if done:
    #             state = env.reset()
    #             print("Saved Video")
    #             recorder.close()
    #             recorder.enabled = False


        

    def get_random_samples(self, replay_buffer, scramble_count, sample_cube_count):
        for idx in range(sample_cube_count):
            self.reset()
            # for _ in range(scramble_count):
            #     self.step()
            self.randomize(scramble_count = scramble_count)
            replay_buffer.append(self.state)


