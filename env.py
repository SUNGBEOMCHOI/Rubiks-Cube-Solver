import gym

def make_env(device, cube_size):
    env_name = 'cube-v0'
    env = gym.make(env_name, cube_size = cube_size, device = device)
    return env