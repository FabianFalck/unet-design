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
    parser = argparse.ArgumentParser(description='Multi-resolution diffusion hyperparameters.')

    # Hyperparams ---
    ## Hyperparams - most used
    parser.add_argument('--USER', type=str, default='user')
    parser.add_argument('--WANDB_MODE', type=str, default="online", metavar='N', help="mode of wandb run tracking, either no tracking ('disabled') or with tracking ('online')")
    parser.add_argument('--DEVICE', type=str, default='cpu', metavar='N', help="device chosen, one in {'cpu','cuda'}. If using less than all GPUs is desired, use CUDA_VISIBLE_DEVICES=0 for instance before your python command.")
    parser.add_argument('--MODEL', type=str, default='unet', metavar='N', help="model chosen, one in {'unet','mlp','unet_wavelet_enc'}.")
    parser.add_argument('--DATASET', type=str, default='mnist', metavar='N', help="dataset chosen, one in {'mnist','celeba'}.")

    parser.add_argument('--to_square_preprocess', type=str2bool, nargs='?', dest='to_square_preprocess', const=True, default=False, help='whether to preprocess MNIST data from triangular to square or not')

    ## Hyperparams - optimization
    parser.add_argument('--SEED', type=int, default=5)
    parser.add_argument('--LR', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--BATCH_SIZE', type=int, default=128)
    parser.add_argument('--NUM_ITERATIONS_LIST', type=int, nargs='*', default=[(10 ** 4)], help='Number of iterations. If list is passed: Activates multi-resolution learning, and values correspond to number of iterations with 1, 2, 3, ... resolutions during multi-resolution learning.')

    ## Hyperparams - diffusion
    # parser.add_argument('--weight_share_res', type=str2bool, nargs='?', dest='weight_share_res', const=True, default=False, help='whether to use the same (high-resolution) model for all resolutions or not')
    parser.add_argument('--RESOLUTION', type=int, default=32, help='Resolution of the data.')  # add "nargs='*'" if you want to pass a list of values"
    parser.add_argument('--BETA_MIN', type=float, default=0.1, help='TODO.')
    parser.add_argument('--BETA_MAX', type=int, default=20, help='Attention: Has to be < N.')
    parser.add_argument('--N', type=int, default=30, help='TODO.')
    parser.add_argument('--EPS', type=float, default=1e-3, help='TODO.')
    parser.add_argument('--T', type=float, default=1.0, help='TODO.')

    ## Hyperparams - MLP Score Network
    parser.add_argument('--ENCODER_LAYERS', type=int, nargs='*', default=[16], help='Latent layer dims of x and t embedding.')
    parser.add_argument('--POS_DIM', type=int, default=16, help='Governs output dim of x and t embedding.')
    parser.add_argument('--DECODER_LAYERS', type=int, nargs='*', default=[128, 128], help='Latent layers dims of network mapping x and t embeddings to noise estimate.')

    ## Hyperparameters - U-Net and G-Net
    parser.add_argument('--NUM_CHANNELS', type=int, default=32, help="Channel 'base multiplier' in U-Net.")
    parser.add_argument('--DROPOUT', type=float, default=0.0, help="Dropout rate in U-Net.")
    parser.add_argument('--NUM_RES_BLOCKS', type=int, default=2, help="Number of blocks in U-Net.")
    parser.add_argument('--AVG_POOL_DOWN', type=str2bool, nargs='?', dest='AVG_POOL_DOWN', const=True, default=False, help='whether to use average pooling or strided convolutions for downsampling in the U-Net')
    # special arguments ---
    parser.add_argument('--DWT_ENCODER', type=str2bool, nargs='?', dest='DWT_ENCODER', const=True, default=False, help='whether to use the DWT encoder or not')
    parser.add_argument('--MULTI_RES_LOSS', type=str2bool, nargs='?', dest='MULTI_RES_LOSS', const=True, default=False, help='whether to use the multi-resolution loss or not')
    parser.add_argument('--FREEZE_LOWER_RES', type=str2bool, nargs='?', dest='FREEZE_LOWER_RES', const=True, default=False, help='freeze parameters of all except the finest level used in decoder in this stage of the sequential training algorithm')
    parser.add_argument('--MODEL_OUT_PASSED_ON', type=str2bool, nargs='?', dest='MODEL_OUT_PASSED_ON', const=True, default=False, help='whether to pass on the model_out prediction directly in the hidden state (True), or keep the tail separate (False)')
    parser.add_argument('--STAGED_PARTITIONED_TIME_INTERVALS', type=str2bool, nargs='?', dest='STAGED_PARTITIONED_TIME_INTERVALS', const=True, default=False, help='whether to use partitioned time intervals during staged training or not.')
    parser.add_argument('--DO_SUPERRES', type=str2bool, nargs='?', dest='DO_SUPERRES', const=True, default=False, help='whether to do ruperresolution or not')

    # Training configs
    parser.add_argument('--TRAIN_ID', type=str, default=None, help="Wandb Run ID (e.g. '3cmfeilu') to continue training of. If None, training a new model from scratch or testing.")
    parser.add_argument('--TRAIN_ITER', type=int, default=None, help="If `train_id` is specified: The iteration to continue training from. Must have saved model for this iteration. If None, the last saved model state is used to continue training.")
    # Testing configs
    parser.add_argument('--TEST_ID', type=str, default=None, help="Wandb Run ID (e.g. '3cmfeilu') to evaluate on test set. If None, we train a model instead.")
    parser.add_argument('--TEST_ITER', type=int, default=None, help="If `test_id` not None, `test_iter` specifies the model state to test. If None, the last saved model state is used for testing.")
    





    ## Hyperparams - logging
    parser.add_argument('--TRAIN_METRICS_EVERY_ITERS', type=int, default=200)
    parser.add_argument('--SAMPLES_EVERY_ITERS', type=int, default=500)
    parser.add_argument('--SAMPLES_EVERY_ITERS_FINAL_RES', type=int, default=-1, help='active if != -1')
    parser.add_argument('--ITERS_PER_MODEL_SAVE', type=int, default=100000)
    parser.add_argument('--SUPERRES_EVERY_ITERS', type=int, default=500, help="Superresolution sampling every iters, if using sequential algorithm and multi-resoution loss ('entire model is trained throughout').")
    parser.add_argument('--WEIGHTED_MULTI_RES_LOSS', type=str2bool, nargs='?', dest='WEIGHTED_MULTI_RES_LOSS', const=True, default=False, help='whether to use the weighted multi-resolution loss or not')
    parser.add_argument('--LOSS_LINEAR_LOOP_IN', type=str2bool, nargs='?', dest='LOSS_LINEAR_LOOP_IN', const=True, default=False, help='whether to loop in new losses in MULTI_RES_LOSS and SEQU_TRAIN_ALGO mode or not')
    parser.add_argument('--U_NET_NORM', type=str2bool, nargs='?', dest='U_NET_NORM', const=True, default=False, help='whether to plot the norms in the U-Net or not')
    parser.add_argument('--U_NET_NORM_N_BATCHES', type=int, default=10, help='Number of batches to compute norms over.')
    parser.add_argument('--U_NET_NORM_EVERY_ITERS', type=int, default=500, help='Norm plot every number of iters. Note: we need the boolean hyperparam as well because it is used by the forward(...) calls.')

    # TODO more



    args = parser.parse_args()
    H = Hyperparams(args.__dict__)

    return H


def check_hyperparams(H):

    # required due to the way discrete_betas is constructed, as there, beta_max divided by N
    assert H.BETA_MAX < H.N

    if H.train_id is not None and H.test_id is not None:
        raise Exception("Restoring training from a specific run and testing a specific run at the same time is not possible.")