import ray

from src.base_rl.trajectory import Trajectory


@ray.remote
class Evaluator:

    def __init__(self, env_creator, env_kwargs, policy_class, policy_kwargs):
        self.env_creator = env_creator
        self.env_kwargs = env_kwargs
        self.policy_class = policy_class
        self.policy_kwargs = policy_kwargs

    def evaluate(self, epochs, render=False):
        env = self.env_creator(**self.env_kwargs)
        policy = self.policy_class(**self.policy_kwargs)
        trajectories = []
        for i in range(epochs):
            trajectory = Trajectory()
            obs = env.reset()
            done = False
            while not done:
                action = policy(obs)
                next_obs, reward, done, info = env.step(action)
                if render:
                    env.render()
                trajectory.add_sample(obs=obs, next_obs=next_obs, reward=reward, action=action)
                obs = next_obs
            trajectories.append(trajectory)

        return trajectories
