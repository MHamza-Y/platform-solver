import os

import dill


def pickle_obj(obj, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        dill.dump(obj, f)


def load_obj(save_path):
    with open(save_path, 'rb') as f:
        return dill.load(f)
