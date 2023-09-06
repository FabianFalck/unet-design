from .unet import UNetModel
from .basic import ScoreNetwork


def get_unet(image_size, image_channels, num_channels=32, dropout=0.0, num_res_blocks = 2):
    num_heads = 4
    num_heads_upsample = -1
    attention_resolutions = "168"
    use_checkpoint = False
    use_scale_shift_norm = True

    # Comment: Channel multiplier governs the factor by which the
    # number of channels in the U-Net are multiplied, and also the number
    # of layers the U-Net has.
    if image_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    elif image_size == 32:
        channel_mult = (2, 2, 2, 2)
    elif image_size == 28:
        channel_mult = (1, 2, 2)
    # added
    elif image_size == 16:
        channel_mult = (1, 2, 2, 2)
    elif image_size == 8:
        channel_mult = (1, 2, 2)
    elif image_size in [4, 2]:
        channel_mult = (1, 2)
    elif image_size == 1: 
        channel_mult = (1,)
    else:
        raise ValueError(f"unsupported image size: {image_size}")

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    kwargs = {
        "in_channels": image_channels,
        "model_channels": num_channels,
        "out_channels": image_channels,
        "num_res_blocks": num_res_blocks,
        "attention_resolutions": tuple(attention_ds),
        "dropout": dropout,
        "channel_mult": channel_mult,
        "num_classes": None,
        "use_checkpoint": use_checkpoint,
        "num_heads": num_heads,
        "num_heads_upsample": num_heads_upsample,
        "use_scale_shift_norm": use_scale_shift_norm,
    }
    return UNetModel(**kwargs)


def get_mlpnet():
    kwargs = {
        "encoder_layers": [16],
        "pos_dim": 16,
        "decoder_layers": [128, 128],
        "x_dim": 2,
    }
    return ScoreNetwork(**kwargs)




