import gym
from ray.tune.registry import register_env
import gym_platform


def env_creator(env_config):
    """
    Function that returns an instantiated instance of the platform environment
    :param env_config: The parameter used by RlLib to pass extra initialization parameters to the environment
    :return: the platform environment object
    """
    return gym.make('Platform-v0')


def register_platform_env(env_name):
    """
    Calling this function registers the platform environment for RlLibs usage
    :param env_name: this name is used by the RlLib to find the registered environment
    """
    register_env(env_name, env_creator)

