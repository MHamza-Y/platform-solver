
import argparse
from src.base_rl.evaluation import Evaluator
from src.envs.creator import env_creator
from src.rllib_utills.policies_wrapper import Policy

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--checkpoint', dest='CHECKPOINT_PATH')
parser.add_argument('-e', '--epochs', dest='EPOCHS', default=100)
args = parser.parse_args()


def visualize_episodes():
    """ Calls an evaluator with render=True using the parsed arguments"""
    env_kwargs = dict(env_config='')
    tmp_env = env_creator(**env_kwargs)
    policy_kwargs = dict(checkpoint_path=args.CHECKPOINT_PATH, obs_space=tmp_env.observation_space)
    evaluator = Evaluator(env_creator=env_creator, env_kwargs=env_kwargs, policy_class=Policy,policy_kwargs=policy_kwargs)
    evaluator.evaluate(epochs=args.EPOCHS,render=True)


visualize_episodes()
