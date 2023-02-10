import gym
import os
import pathlib
import envs.creator
from rllib_utills.policies_wrapper import Policy


def evaluate(epochs=100):
    env_name = "Platform-v0"
    env = gym.make(env_name)

    path = max(pathlib.Path('tmp/ray_exp_logs/platform_solver').glob('**/checkpoint_0*'), key=os.path.getmtime)
    print(path)
    policy = Policy(path, env.observation_space)

    for _ in range(epochs):
        episode_reward = 0
        done = False
        obs = env.reset()
        while not done:
            action = policy(obs)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            env.render()


evaluate()
