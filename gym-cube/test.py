from typing import List
import gym_cube
import gym
import os
import time
import numpy as np
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import moviepy
from typing import List
from utils import ReplayBuffer
import torch

'''
action
UDFBRLU'D'F'B'R'L'
01234567891011
'''
e = gym.make("cube-v0", render_mode = 'rgb_array')
e.reset()

# reward check

# s0, r0 ,d0= e.step(0)
# print(r0, d0) # -1, 0
# s1, r1, d1 = e.step(6)
# print(r1, d1) # 1, 1

# get random sample check

# device_ver = 'cpu'
# device = torch.device('cuda' if device_ver=='cuda' and torch.cuda.is_available() else 'cpu')
# buffer_size = 10
# sample_scramble_count = 5
# sample_cube_count = 2
# replay_buffer = ReplayBuffer(buffer_size)
# e.get_random_samples(replay_buffer, sample_scramble_count, sample_cube_count)
# print(len(replay_buffer))

# state check

# cube = e.cube
# icube = e.icube
# stickers = icube._project(cube._stickers)[:,:,:2]
# sticker_centroids = icube._project(cube._sticker_centroids)[:,:3]
# print(len(stickers)) # 54
# print(stickers[:9]) # 9 x 2 x 54 -> 각각의 cubelet face 당 9 x 2의 정보가 들어있다
# print(stickers[45:54])
# e.step(0)
# print(len(stickers[45:54]))

# check = 0
# for i in range(6):
#     before = e.icube._project(e.cube._stickers)[:,:,2][9*(i): 9*(i+1)]
#     e.step(0)
#     after = e.icube._project(e.cube._stickers)[:,:,2][9*(i): 9*(i+1)]
#     for _ in range(9):
#         check += np.array_equal(before[_], after[_])
#         if check == 0:
#             print("face index: ", i)
#             print("cubelet index: ", _)


# print(sticker_centroids) # 3 x 18
# print(sticker_centroids[:,2]) # 1 x 3



# video check

# e.randomize(scramble_count=10)
# e.render()
# curr_dir = os.path.dirname(__file__)
# video_path = curr_dir + '/video.mp4'
# # recorder = VideoRecorder(env = e, path = video_path)
# frames_per_sec = 60
# recorded_frames = []
# render_history = []
# frame = e.render()
# # print(type(frame))
# # if isinstance(frame, List):
# #     render_history += frame
# #     frame = frame[-1]
# # else:
# #     print("Error")
# # recorded_frames.append(frame)

# # 켜져있는 상태에서 녹화를 할 수는 없을까?

# num_actions = 20
# i = 0
# # recorder.capture_frame()
# face_list = ["U", "D", "L", "R", "B", "F"]
# while i < num_actions:
#     frame = e.render()
#     # a = np.random.randint(12)
#     # e.step(a)
#     face = face_list[np.random.randint(6)]
#     layer = np.random.randint(3)
#     e.icube.rotate_face(face, 1, layer, 1)
#     recorded_frames.append(frame)
#     i += 1
# print("Saved Video")
# # recorder.close()
# # recorder.enabled = False
# e.close()
# if len(recorded_frames) > 0:
#     from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

# clip = ImageSequenceClip(recorded_frames, fps = frames_per_sec)
# clip.write_videofile(video_path)

