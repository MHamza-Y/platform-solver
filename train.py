from ray.rllib.algorithms.ppo import PPOConfig
from ray import air, tune
import envs.creator


def train_policy():
    total_workers = 12
    num_envs_per_worker = 24

    config = (
        PPOConfig()
        .environment('Platform-v0')
        .training(gamma=0.995, num_sgd_iter=10, sgd_minibatch_size=1000, clip_param=0.1, lr=1e-4, train_batch_size=2000,
                  entropy_coeff=1e-4)
        .resources(num_gpus=1, num_cpus_per_worker=1)
        .rollouts(num_rollout_workers=total_workers, num_envs_per_worker=num_envs_per_worker)
        .framework("torch")
        .training(
            model={"fcnet_hiddens": [64, 64, 64], "vf_share_layers": False, "use_lstm": True, "lstm_cell_size": 32,
                   "max_seq_len": 5})
        .evaluation(evaluation_num_workers=1)
    )
    tuner = tune.Tuner(
        "PPO",
        param_space=config,
        run_config=air.RunConfig(
            name="platform_solver",
            local_dir="tmp/ray_exp_logs",
            sync_config=tune.SyncConfig(
                syncer=None  # Disable syncing
            ),
            stop={"training_iteration": 140, "episode_reward_mean": 0.99},
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=5,
                checkpoint_score_order='max',
                checkpoint_score_attribute="episode_reward_mean",
                num_to_keep=3,
                checkpoint_at_end=True
            )
        ),

    )

    tuner.fit()


train_policy()
