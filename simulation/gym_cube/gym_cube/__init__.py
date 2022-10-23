from gym.envs.registration import register


register(
    id='cube-v0',
    entry_point='gym_cube.envs:CubeEnv',
)
