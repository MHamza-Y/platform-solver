import gym
from ray.tune.registry import register_env
import gym_platform


def env_creator(env_config):
    return gym.make('Platform-v0')


def register_platform_env(env_name):
    register_env(env_name, env_creator)


#register_env("Platform-v0", env_creator)
