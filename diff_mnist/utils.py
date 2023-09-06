
import argparse
import yaml
import wandb
import os
import torch

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


def str2bool(v):
    """
    Source code copied from: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def download_some_wandb_files(files_to_restore, run_id):

    api = wandb.Api()
    wandb_args = load_dict_from_yaml('setup/wandb.yml')
    run_path = wandb_args['team_name'] + "/" + wandb_args['project_name'] + "/" + run_id
    run = api.run(run_path)
    download_dir = wandb.run.dir

    os.chdir(download_dir)

    # print("Download directory: ", os.getcwd())

    # restore the files files
    for file_path in files_to_restore:
        # run.file(file_path).download()   # is an alternative, should also work
        wandb.restore(file_path, run_path=run_path)

    os.chdir('../../../')


def compute_norm(tensor):
    """
    Compute the norm of a tensor
    """
    # multiple the dimensions, starting from the first one 
    dim = 1
    for sub_dim in tensor.shape[1:]:
        dim *= sub_dim
    norm_factor = 1 / dim
    return torch.linalg.norm(torch.flatten(tensor, start_dim=1), dim=1) * norm_factor