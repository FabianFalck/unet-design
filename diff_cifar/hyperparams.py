from utils import str2bool
import argparse


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


def args_parser():

    # 30/3/23: Note: the below config is cross-checked with the original configs file in config/. the default parameters below correspond to the contents in it.

    parser = argparse.ArgumentParser(description='Multi-resolution diffusion hyperparameters.')

    parser.add_argument('--SEED', type=int, default=1)

    # UNet
    parser.add_argument('--ch', type=int, default=128, metavar='N', help="base channel of UNet")
    parser.add_argument('--ch_mult', type=int, nargs='+', default=[1, 2, 2, 2], metavar='N', help="channel multiplier")
    parser.add_argument('--attn', type=int, nargs='+', default=[1], metavar='N', help="add attention to these levels")
    parser.add_argument('--num_res_blocks', type=int, default=2, metavar='N', help="# resblock in each level")
    parser.add_argument('--dropout', type=float, default=0.1, metavar='N', help="dropout rate of resblock")
    parser.add_argument("--DOWNSAMPLE_TYPE", type=str, default="conv", metavar='N', help="downsample type. possible values are ['conv', 'avg_pool'].")
    # Gaussian Diffusion
    parser.add_argument('--beta_1', type=float, default=1e-4, metavar='N', help="start beta value")
    parser.add_argument('--beta_T', type=float, default=0.02, metavar='N', help="end beta value")
    parser.add_argument('--T', type=int, default=1000, metavar='N', help="total diffusion steps.")
    parser.add_argument('--mean_type', type=str, default='epsilon', metavar='N', help="predict variable. possible values are ['xprev', 'xstart', 'epsilon'].")
    parser.add_argument('--var_type', type=str, default='fixedlarge', metavar='N', help="variance type. possible values are ['fixedlarge', 'fixedsmall'].")
    # Training
    parser.add_argument('--lr', type=float, default=2e-4, metavar='N', help="target learning rate")
    parser.add_argument('--grad_clip', type=float, default=1., metavar='N', help="gradient norm clipping")
    parser.add_argument('--img_size', type=int, default=32, metavar='N', help="image size")
    parser.add_argument('--warmup', type=int, default=5000, metavar='N', help="learning rate warmup")
    parser.add_argument('--batch_size', type=int, default=128, metavar='N', help="batch size")
    parser.add_argument('--num_workers', type=int, default=4, metavar='N', help="workers of Dataloader")
    parser.add_argument('--ema_decay', type=float, default=0.9999, metavar='N', help="ema decay rate")
    parser.add_argument('--parallel', type=str2bool, default=False, metavar='N', help="multi gpu training")
    # Logging & Sampling
    # parser.add_argument('--logdir', type=str, default='./logs/' + get_timestamp(), metavar='N', help="log directory")

    # restore training
    parser.add_argument('--TRAIN_ID', type=str, default=None, metavar='N', help="Train id whose run shall be restored.")
    parser.add_argument('--TRAIN_ITER', type=int, default=None, metavar='N', help="Iteration from which to continue training. If None is provided, the latest iteration is used.")

    # restore test
    parser.add_argument('--TEST_ID', type=str, default=None, metavar='N', help="Test id whose run shall be tested.")
    parser.add_argument('--TEST_ITER', type=int, default=None, metavar='N', help="Iteration at which to evaluate.")

    parser.add_argument('--sample_size', type=int, default=64, metavar='N', help="sampling size of images")
    parser.add_argument('--sample_step', type=int, default=10000, metavar='N', help="frequency of sampling")
    parser.add_argument('--TRAIN_METRICS_EVERY_ITERS', type=int, default=200, metavar='N', help="frequency of logging training metrics (loss), in #iterations")
    # Evaluation
    parser.add_argument('--save_step', type=int, default=30000, metavar='N', help="frequency of saving checkpoints, 0 to disable during training")
    parser.add_argument('--eval_step', type=int, default=200000, metavar='N', help="frequency of evaluating model, 0 to disable during training")
    parser.add_argument('--num_images', type=int, default=5000, metavar='N', help="the number of generated images for evaluation")
    parser.add_argument('--fid_use_torch', type=str2bool, default=False, metavar='N', help="calculate IS and FID on gpu")
    parser.add_argument('--fid_cache', type=str, default='./stats/cifar10.train.npz', metavar='N', help="FID cache")
    # new
    parser.add_argument('--user', type=str, default='user', metavar='N', help="user name")
    parser.add_argument('--device', type=str, default='cpu', metavar='N', help="device to use for training")
    parser.add_argument('--model', type=str, default='unet', metavar='N', help="model to use for training")
    parser.add_argument('--WANDB_MODE', type=str, default='online', metavar='N', help="wandb mode")

    # special arguments
    parser.add_argument('--NUM_ITERATIONS_LIST', type=int, nargs='+', default=[1200005], metavar='N', help='Number of iterations. If list is passed: Activates multi-resolution learning, and values correspond to number of iterations with 1, 2, 3, ... resolutions during multi-resolution learning.')
    parser.add_argument('--DWT_ENCODER', type=str2bool, default=False, metavar='N', help='whether to use the DWT encoder or not')
    parser.add_argument('--FREEZE_LOWER_RES', type=str2bool, default=False, metavar='N', help='freeze parameters of all except the finest level used in decoder in this stage of the sequential training algorithm')
    parser.add_argument('--MULTI_RES_LOSS', type=str2bool, default=False, metavar='N', help='whether to use the multi-resolution loss or not')

    args = parser.parse_args()
    FLAGS = Hyperparams(args.__dict__)

    return FLAGS




def check_hyperparams(H):


    if H.TRAIN_ID is not None and H.TEST_ID is not None:
        raise Exception("Restoring training from a specific run and testing a specific run at the same time is not possible.")