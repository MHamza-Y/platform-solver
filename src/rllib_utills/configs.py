import os
import pickle


def get_checkpoint_configs(checkpoint):
    run_base_dir = os.path.dirname(checkpoint)
    config_path = os.path.join(run_base_dir, 'params.pkl')
    with open(config_path, 'rb') as f:
        configs = pickle.load(f)
    return configs
