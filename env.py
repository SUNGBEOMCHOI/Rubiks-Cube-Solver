from collections import namedtuple


import gym

def make_env(env_name, cube_size=3):
    """
    Make gym environment
    Args:
        env_name: Environment name you want to make
        cfg: config information which contains cube size
    Returns:
        env: gym environment
    """
    # TODO: gym make 할때 cube size를 넣어서 큐브를 생성
    env = gym.make(env_name, cube_size)
    return 

class Cube(gym.Env):
    def __init__(self, cube_size=3):
        """
        Gym environment for cube

        Args:
            cube_size: Cube size you want to make environment
        """
        super().__init__()
        self.transaction = namedtuple('Point', ['state', 'target_value', 'target_policy', 'scramble_count'])
        pass

    def reset(self, seed=None, scramble_count=1000):
        """
        Reset the state to random scrambled cube

        Args:
            seed: Random seed number
            scramble_count: Number of scramble cubes randomly

        Return:
            Initial state shape [number of cublets, possible locations]
        """
        pass

    def step(self, action):
        """
        Execute one time step within the environment

        Args:
            action: Action you want to perform

        Returns:
            state: Numpy array of state after action is performed
            reward: Return +1 if state is goal state, else return -1
            done: Return true if state is goal state, else return false
            info: Dictionary of useful information
        """
        pass

    def render(self, mode):
        """
        Render the environment to the screen

        Args:
            mode:
        """
        pass

    def save_video(self, video_path):
        """
        Save playing video to specific path

        Args:
            video_path: Path to save video
        """
        pass

    def get_random_samples(self, replay_buffer, model, scramble_count, sample_cube_count):
        """
        Return samples which contain state, target value, target policy for training
        
        Args:
            replay_buffer: Replay buffer to save samples
            scramble_count: Number of scramble cubes randomly
            sample_cube_count: Number of cube samples

        Returns:
            List of samples of namedtuple type
            Number of total size is scramble_count*sample_cube_count
        """
        pass
