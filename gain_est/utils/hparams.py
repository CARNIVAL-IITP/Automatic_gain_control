import json, os
import yaml


class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v
    
    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def update(self, kwargs):
        for k, v in kwargs.items():
            self[k] = v
    
    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()


def get_hparams(config_dir:str, base_dir:str = "", save:bool = False) -> HParams:
    '''1. load .json or .yaml file and return HParams
    2. if save=True, save file'''
    with open(config_dir, 'r') as f:
        data = f.read()
    
    if config_dir.endswith(".json"):
        config = json.loads(data)
        save_dir = 'config.json'
    elif config_dir.endswith(".yaml"):
        config = yaml.safe_load(data)
        save_dir = 'config.yaml'
    
    if save:
        with open(os.path.join(base_dir, save_dir), 'w') as f:
            f.write(data)

    hps = HParams(**config)
    hps.base_dir = base_dir

    return hps
