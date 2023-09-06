from argparse import Namespace
from functorch import vmap
from itertools import repeat


def namespace_to_dict(ns, copy=True):
    d = vars(ns)
    if copy:
        d = d.copy()
    return d


class DataClass(Namespace):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def items(self):
        return namespace_to_dict(self).items()

    def __getitem__(self, key):
        return self.__getattribute__(key)

# from documentation: https://pytorch.org/tutorials/prototype/vmap_recipe.html
# "The first use case for vmap is making it easier to handle batch dimensions in your code.
# One can write a function func that runs on examples and then lift it to a function that can take batches of examples with vmap(func). f
batch_mul = vmap(lambda x, y: x * y)


def repeater(data_loader):
    for loader in repeat(data_loader):
        for data in loader:
            yield data


def load_dict_from_yaml(file_path):
    """
    Load args from .yml file.
    """
    with open(file_path) as file:
        yaml_dict = yaml.safe_load(file)

    # create argsparse object
    # parser = argparse.ArgumentParser(description='MFCVAE training')
    # args, unknown = parser.parse_known_args()
    # for key, value in config.items():
    #     setattr(args, key, value)

    # return args.__dict__  # return as dict, not as argsparse
    return yaml_dict