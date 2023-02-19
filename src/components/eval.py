import pandas as pd

from src.base_rl.evaluation import EvaluationActor
from src.rllib_utills.policies_wrapper import Policy
import warnings


def get_evenly_divided_values(value_to_be_distributed, times):
    return [value_to_be_distributed // times + int(x < value_to_be_distributed % times) for x in range(times)]


def eval_env(ray, epochs, workers, env_creator, env_kwargs, checkpoint_path, render=False):
    if render:
        warnings.warn('render=True, numbers of workers changed to 1')
        workers = 1
    divided_epochs = get_evenly_divided_values(epochs, workers)
    tmp_env = env_creator(**env_kwargs)
    policy_kwargs = dict(checkpoint_path=checkpoint_path, obs_space=tmp_env.observation_space)

    evaluators = [EvaluationActor.remote(env_creator=env_creator, env_kwargs=env_kwargs, policy_class=Policy,
                                   policy_kwargs=policy_kwargs) for _ in range(workers)]

    evaluators_futures = [evaluator.evaluate.remote(epochs=divided_epochs[i], render=False) for i, evaluator in
                          enumerate(evaluators)]

    evaluator_results = ray.get(evaluators_futures)
    evaluator_results = sum(evaluator_results, [])
    result_dfs = [result.get_dataframe(ep_id=i) for i, result in enumerate(evaluator_results)]

    results_df = pd.concat(result_dfs, axis=0)
    return results_df
