from functools import wraps
from typing import Any, Callable, Dict, Mapping, Sequence, Union

import numpy as np
import torch
from torch import Tensor
from pytorch_lightning.utilities import move_data_to_device


def _has_method(o: object, name: str) -> bool:
    return callable(getattr(o, name, None))


def auto_move_data(fn: Callable) -> Callable:
    """Decorator for ``LightningModule`` methods for which input args should be moved to the correct device.

    Typically, applied to ``__call__`` or ``forward``.

    Args:
        fn: A LightningModule method for which the arguments should be moved to the device the parameters are on.

    Example::

        # directly in the source code
        class LitModel(LightningModule):

            @auto_move_data
            def forward(self, x):
                return x

        # or outside
        LitModel.forward = auto_move_data(LitModel.forward)

        model = LitModel()
        model = model.to('cuda')
        model(torch.zeros(1, 3))

        # input gets moved to device
        # tensor([[0., 0., 0.]], device='cuda:0')

    """

    @wraps(fn)
    def auto_transfer_args(self, *args, **kwargs):
        args, kwargs = move_data_to_device((args, kwargs), self.device)
        return fn(self, *args, **kwargs)

    return auto_transfer_args
