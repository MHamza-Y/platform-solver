from ray import air, tune


def tune_hyper_param(algo, config, log_dir, iterations,
                     name,max_concurrent_trials=1):
    tuner = tune.Tuner(
        algo,
        param_space=config,
        tune_config=tune.TuneConfig(
            metric="episode_reward_mean",
            mode="max",
            max_concurrent_trials=max_concurrent_trials
        ),
        run_config=air.RunConfig(
            name=name,
            local_dir=log_dir,

            stop={"training_iteration": iterations},
        ),

    )

    results = tuner.fit()
    return results
