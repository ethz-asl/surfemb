from collections import defaultdict
import yaml


class DatasetConfig:
    model_folder = 'models'
    train_folder = 'train'
    test_folder = 'test'
    img_folder = 'rgb'
    depth_folder = 'depth'
    img_ext = 'png'
    depth_ext = 'png'

    @classmethod
    def from_yaml(cls, yaml_file_path: str):
        with open(yaml_file_path, "r") as f:
            cfg_yaml = yaml.load(f, Loader=yaml.SafeLoader)

        cfg = DatasetConfig()

        assert (sorted(cfg_yaml.keys()) == sorted([
            'model_folder', 'train_folder', 'test_folder', 'img_folder',
            'depth_folder', 'img_ext', 'depth_ext'
        ]))
        for key in cfg_yaml.keys():
            setattr(cfg, key, cfg_yaml[key])

        return cfg


config = defaultdict(lambda *_: DatasetConfig())

config['tless'] = tless = DatasetConfig()
tless.model_folder = 'models_cad'
tless.test_folder = 'test_primesense'
tless.train_folder = 'train_primesense'

config['hb'] = hb = DatasetConfig()
hb.test_folder = 'test_primesense'

config['itodd'] = itodd = DatasetConfig()
itodd.depth_ext = 'tif'
itodd.img_folder = 'gray'
itodd.img_ext = 'tif'
