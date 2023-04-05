from . import decoders
from . import utils

from .decoders.unetplusplus import UnetPlusPlus
from .decoders.fpn import FPN

from typing import Optional as _Optional
import torch as _torch


def create_model(
    arch: str,
    encoder_name: str = "resnet34",
    encoder_weights: _Optional[str] = "imagenet",
    in_channels: int = 3,
    classes: int = 1,
    **kwargs,
) -> _torch.nn.Module:

    archs = [
        FPN,
        UnetPlusPlus,
    ]
    archs_dict = {a.__name__.lower(): a for a in archs}
    try:
        model_class = archs_dict[arch.lower()]
    except KeyError:
        raise KeyError(
            "Wrong architecture type `{}`. Available options are: {}".format(
                arch,
                list(archs_dict.keys()),
            )
        )
    return model_class(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        **kwargs,
    )
