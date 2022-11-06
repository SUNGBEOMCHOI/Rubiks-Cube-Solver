import os
import sys
import math
import gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from collections import namedtuple
from assets.cube_interactive import Cube as RenderCube
from assets.py222.py222 import initState, getOP, doMove, isSolved, getStickers, printCube
# from assets.PyCuber.pycuber.cube import Cube, Cubie, Centre, Corner, Edge, Square, Step, Formula
import pycuber as pc
from utils import *
from assets.cube_utils import isSolved_
import copy

class CubeEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array']}
    def __init__(self, cube_size = 2, device = 'cpu'):
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
        self.init_state()
        self.frames = []
    
    def init_state(self):
        if self.cube_size == 2:
            self.sim_cube = initState() # py222
            self.cube = self.sim_state_to_state(self.sim_cube)
        elif self.cube_size == 3:
            self.sim_cube = pc.Cube()
            self.cube = self.sim_state_to_state(self.sim_cube)
        else:
            raise NotImplementedError
        if self.show_cube:
            self.render_cube = RenderCube(self.cube_size)
            self.fig = self.render_cube.draw_interactive()        

    def reset(self, seed = None, scramble_count = 2):
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
        info = {}
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
            # if isSolved_(self.sim_cube):
            if isSolved_(self.sim_cube.perform_step(sim_action)):
                done = True
                reward = 1.0
            else:
                done = False
                reward = -1.0
            self.cube = self.sim_state_to_state(self.sim_cube)
        else:
            raise NotImplementedError
        
        if self.show_cube:
            face, degree = self.action_to_sim_action['render'][action] # face string , degree number
            self.fig.axes[3].rotate_face(face, degree, layer = 0)
        return self.cube, reward, done, info

    def render(self):
        self.render_cube = RenderCube(self.cube_size)
        self.fig = self.render_cube.draw_interactive()
        self.show_cube=True
        return self.fig
    
    def close_render(self):
        self.show_cube = False
    
    def sim_state_to_state(self, sim_state):
        if self.cube_size == 2:
            state = np.zeros(self.state_dim)
            for position, cubelet in enumerate(getOP(sim_state)):
                state_cubelet, position_idx = cubelet
                state_position = position * 3 + position_idx
                state[state_cubelet][state_position] = 1.0 # one hot
        elif self.cube_size == 3:
            corner_location_list = ["LDB", "LDF", "LUB", "LUF", "RDB", "RDF", "RUB", "RUF"]
            edge_location_list = ["LB", "LF", "LU", "LD", "DB", "DF", "UB", "UF", "RB", "RF", "RU", "RD"]
            corner_state = np.zeros([8, 24])
            edge_state = np.zeros([12, 24])
            corner_list = []
            edge_list = []
            u_list = sim_state.at_face("U") 
            d_list = sim_state.at_face("D")
            f_list = sim_state.at_face("F")
            b_list = sim_state.at_face("B")

            for u in u_list:
                if u.type == 'corner':
                    corner_list.append(u)
                elif u.type == 'edge':
                    edge_list.append(u)
            for d in d_list:
                if d.type == 'corner':
                    corner_list.append(d)
                elif d.type == 'edge':
                    edge_list.append(d)
            for f in f_list:
                if f.type == 'edge':
                    if f.location == 'LF' or f.location == 'RF':
                        edge_list.append(f)
            for b in b_list:
                if b.type == 'edge':
                    if b.location == 'LB' or b.location == 'RB':
                        edge_list.append(b)

            # 색 조합으로 위치 찾기
            corner_color_dict = {}
            for corner in corner_list:
                corner_color_dict[corner.location] = [corner.facings, corner.facings[corner.location[1]]] # U, D

            edge_color_dict = {}
            for edge in edge_list:
                if edge.location[0] == 'L':
                    edge_color_dict[edge.location] = [edge.facings, edge.facings[edge.location[1]]] # LU, LD. LF, LB 
                elif edge.location[0] == 'R':
                    edge_color_dict[edge.location] = [edge.facings,edge.facings[edge.location[1]]] # RU, RD. RF. RB
                elif edge.location[0] == 'U':
                    edge_color_dict[edge.location] = [edge.facings,edge.facings[edge.location[0]]] # UB, UF
                elif edge.location[0] == 'D':
                    edge_color_dict[edge.location] = [edge.facings,edge.facings[edge.location[0]]] # DB, DF
                else:
                    pass

            # corner의 색 조합은 유일 -> 같은 색 조합을 갖는 corner를 찾고 X Y Z 에 따라 3가지 경우의 수가 존재함을 이용
            base_cube = pc.Cube()
            for key in corner_color_dict.keys():
                facings, target_face = corner_color_dict[key][0], corner_color_dict[key][1]
                facings_set = set()
                for face_key in facings.keys():
                    facings_set.add(facings[face_key])
                for position, corner_location in enumerate(corner_location_list): # 0 ~ 7 corner location
                    base_color_map = base_cube[corner_location].facings
                    base_set = set()
                    for base_key in base_color_map.keys():
                        base_set.add(base_color_map[base_key])
                    if facings_set == base_set: # finding same color set: 0 ~ 7, 처음 위치와 현재 위치가 다르다
                        for position_idx, base_key in enumerate(base_color_map):
                            base_color = base_color_map[base_key]
                            if target_face == base_color: # finding target face position idx: 0 ~ 2
                                state_position = position * 3 + position_idx
                                corner_state[position][state_position] = 1.0

            for key in edge_color_dict.keys():
                facings, target_face = edge_color_dict[key][0], edge_color_dict[key][1]
                facings_set = set()
                for face_key in facings.keys():
                    facings_set.add(facings[face_key])
                for position, edge_location in enumerate(edge_location_list):
                    base_color_map = base_cube[edge_location].facings
                    base_set = set()
                    for base_key in base_color_map.keys():
                        base_set.add(base_color_map[base_key])
                    if facings_set == base_set: # 0 ~ 11
                        for position_idx, base_key in enumerate(base_color_map):
                            base_color = base_color_map[base_key]
                            if target_face == base_color:
                                state_position = position * 2 + position_idx # 0 ~ 1
                                edge_state[position][state_position] = 1.0
            state = np.concatenate([corner_state, edge_state])
        else:
            raise NotImplementedError
        return state
    
    def state_to_sim_state(self, state):
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
        for sample_cube_idx in range(1, sample_cube_count+1):
            self.init_state()
            action_sequence = np.random.randint(self.action_dim, size=sample_scramble_count)
            for scramble_idx, action in enumerate(action_sequence):
                state, _, _, _ = self.step(action)
                target_value, target_policy, error = self.get_target_value(model, scramble_idx+1, temperature)
                sample = self.transaction(state, target_value, target_policy, scramble_idx+1, error)
                replay_buffer.append(sample)

    def get_target_value(self, model, scramble_count, temperature):
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
                
                # next_sim_cube = self.sim_cube.perform_step(sim_action)
                # next_state = self.sim_state_to_state(next_sim_cube)
                
                if action % 2 == 0:
                    counter_action = action + 1
                else:
                    counter_action = action - 1

                
                # if isSolved_(next_sim_cube):
                if isSolved_(self.sim_cube.perform_step(sim_action)):
                    reward = 1.0
                    target_value, target_policy = 1.0, action
                    break
                else:
                    reward = -1.0

                # next_state = self.sim_state_to_state(self.sim_cube)
                next_state_list.append(self.sim_state_to_state(self.sim_cube))
                self.sim_cube.perform_step(self.action_to_sim_action[self.cube_size][counter_action])
                
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
            error = abs(value.detach().item() - target_value)
        return target_value, target_policy, error
    
    def save_video(self, cube_size, scramble_count, sample_cube_count, video_path='./video'):
        filename = f'cube{cube_size}_scramble{scramble_count}_sample{sample_cube_count}.gif'
        self.frames = self.fig.axes[3].frames # interactivecube.frames
        plt.figure(figsize=(self.frames[0].shape[1] / 72.0, self.frames[0].shape[0] / 72.0), dpi=72)
        patch = plt.imshow(self.frames[0])
        plt.axis('off')

        def animate(i):
            patch.set_data(self.frames[i])

        anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(self.frames), interval=0)
        anim.save(video_path + '/'+ filename, writer='imagemagick', fps=5)
    




    