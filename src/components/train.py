from ray import air, tune


def train_env(algo, config, log_dir, iterations, stop_reward_mean,
              name):
    tuner = tune.Tuner(
        algo,
        param_space=config,
        run_config=air.RunConfig(
            name=name,
            local_dir=log_dir,
            sync_config=tune.SyncConfig(
                syncer=None  # Disable syncing
            ),
            stop={"training_iteration": iterations, "episode_reward_mean": stop_reward_mean},
            checkpoint_config=air.CheckpointConfig(
                checkpoint_at_end=True
            )
        ),

    )
    results = tuner.fit()

    return results
