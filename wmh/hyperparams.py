import argparse

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
    


class Hyperparams(dict):
    """
    Wrapper class for a dictionary required for pickling.
    """
    def __init__(self, input_dict):
        for k, v in input_dict.items():
            self[k] = v

    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            return None

    def __setattr__(self, attr, value):
        self[attr] = value

    # required for pickling !
    def __getstate__(self): return self.__dict__
    def __setstate__(self, d): self.__dict__.update(d)

    

# freeze_lower_res: bool = False,
# num_epochs_list: List[int] = [1000000000], 
# up_fct: str = 'interpolate_nearest', 
# n_extra_resnet_layers: int = 0,
# multi_res_loss: bool = False,
# hidden_channels: int = 64,
# no_skip_connection: bool = False,
# no_down_up: bool = False,
# dwt_mode: str = 'zero',
# dwt_wave: str = 'haar',


def args_parser():

    # 30/3/23: Note: the below config is cross-checked with the original configs file in config/. the default parameters below correspond to the contents in it.

    parser = argparse.ArgumentParser(description='Multi-resolution diffusion hyperparameters.')

    parser.add_argument('--wandb_mode', type=str, default='online', metavar='N')
    parser.add_argument('--device', type=str, default='cpu', metavar='N')

    parser.add_argument('--batch_size', type=int, default=32, metavar='N', help="batch size")
    parser.add_argument('--lr', type=float, default=2e-4, metavar='N', help="target learning rate")
    parser.add_argument('--data_augmentation', type=str, default='none', metavar='N', help="which data augmentation to use, in [none, auto, manual1]")

    parser.add_argument('--dwt_encoder', type=str2bool, default=False, metavar='N')
    parser.add_argument('--freeze_lower_res', type=str2bool, default=False, metavar='N')
    parser.add_argument('--num_epochs_list', type=int, nargs='+', default=[1200005])
    parser.add_argument("--up_fct", type=str, default='interpolate_nearest', metavar='N')
    parser.add_argument('--n_extra_resnet_layers', type=int, default=0, metavar='N')
    parser.add_argument('--multi_res_loss', type=str2bool, default=False, metavar='N')
    parser.add_argument('--hidden_channels', type=int, default=64, metavar='N')
    parser.add_argument('--no_skip_connection', type=str2bool, default=False, metavar='N')
    parser.add_argument('--no_down_up', type=str2bool, default=False, metavar='N')
    parser.add_argument('--dwt_mode', type=str, default='zero', metavar='N')
    parser.add_argument('--dwt_wave', type=str, default='haar', metavar='N')

    parser.add_argument('--train_loss_every_iters', type=int, default=100, metavar='N')
    parser.add_argument('--train_hist_every_iters', type=int, default=1000, metavar='N')
    parser.add_argument('--train_prec_recall_curve_every_iters', type=int, default=1000, metavar='N')
    parser.add_argument('--val_every_epochs', type=int, default=5, metavar='N')
    # parser.add_argument('--val_every_iters', type=int, default=5000, metavar='N')

    parser.add_argument('--n_images_seg_to_plot', type=int, default=75, metavar='N')

    parser.add_argument('--early_stop_patience', type=int, default=10, metavar='N', help='-1 disables early stopping')
    parser.add_argument('--early_stop_min_improvement', type=float, default=0.001, metavar='N', help='default is 0.1% improvement')  

    parser.add_argument('--debug_breaks', type=str2bool, default=False, metavar='N')

    parser.add_argument('--seed', type=int, default=1, metavar='N')

    args = parser.parse_args()
    H = Hyperparams(args.__dict__)

    return H