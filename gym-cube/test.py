import gym_cube
import gym
import os
import time

e = gym.make("cube-v0")
e.reset()
# ts = time.time()
action = ("U", 1, -1)
# e.randomize(seed = None, scramble_count= 60000) # 13.37 sec
# mode = [0, 1]
# e.render_mode(mode)
# e.step(action)
e.randomize(scramble_count=10)
e.reset()
e.render()


# policy = [("U", 1, 1), ("U", 1, 1), ("U", 1, 1), ("U", 1, 1)]
# path = 'gym-cube/result'
# e.save_video(e, policy, path)
# print(time.time() - ts)