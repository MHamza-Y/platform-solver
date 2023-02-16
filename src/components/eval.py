import gym
import pandas as pd

from src.base_rl.evaluation import Evaluator
from src.rllib_utills.policies_wrapper import Policy


def eval_enva(epochs, env_name, checkpoint_path):
    env = gym.make(env_name)
    policy = Policy(checkpoint_path, env.observation_space)

    for _ in range(epochs):
        episode_reward = 0
        done = False
        obs = env.reset()
        while not done:
            action = policy(obs)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            env.render()


def get_evenly_divided_values(value_to_be_distributed, times):
    return [value_to_be_distributed // times + int(x < value_to_be_distributed % times) for x in range(times)]


def eval_env(ray, epochs, workers, env_creator, env_kwargs, checkpoint_path):
    divided_epochs = get_evenly_divided_values(epochs, workers)
    tmp_env = env_creator(**env_kwargs)
    policy_kwargs = dict(checkpoint_path=checkpoint_path, obs_space=tmp_env.observation_space)

    evaluators = [Evaluator.remote(env_creator=env_creator, env_kwargs=env_kwargs, policy_class=Policy,
                                   policy_kwargs=policy_kwargs) for _ in range(workers)]

    evaluators_futures = [evaluator.evaluate.remote(epochs=divided_epochs[i], render=False) for i, evaluator in
                          enumerate(evaluators)]

    evaluator_results = ray.get(evaluators_futures)
    evaluator_results = sum(evaluator_results, [])
    result_dfs = [result.get_dataframe(ep_id=i) for i, result in enumerate(evaluator_results)]

    results_df = pd.concat(result_dfs, axis=0)
    return results_df
