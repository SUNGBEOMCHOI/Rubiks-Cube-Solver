import gym_cube
import gym
import os
import time

e = gym.make("cube-v0")
e.reset()
ts = time.time()
# action = ("U", 1, 1)
# e.step(action)
# mode = [0, 1]
# e.render_mode(mode)
policy = [("U", 1, 1), ("U", 1, 1), ("U", 1, 1), ("U", 1, 1)]
path = 'gym-cube/result'
e.save_video(e, policy, path)
print(time.time() - ts)