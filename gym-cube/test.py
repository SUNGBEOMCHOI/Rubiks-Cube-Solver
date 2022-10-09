import gym_cube
import gym

e = gym.make("cube-v0")
e.reset()
action = ("U", 1, 1)
e.step(action)
mode = [0, 1]
e.render_mode(mode)