import torch

import torch.nn.functional as F


class Differentiation:
    """
    Differentiate augmentation function.

    Parameters
    ----------
    None
    """
    def __init__(self, order=1):
        self.order = order

    def __call__(self, x):
        """
        Perform differentiation augmentation.

        Parameters
        ----------
        x: torch tensor of shape (n_samples, n_channels, seq_len)
            Original time series.
        """
        # Apply differentiation along the time dimension (last dimension)
        x_diff = torch.diff(x, dim=-1, n=self.order)
        x_diff = F.pad(x_diff, (0, self.order), mode="constant", value=0)  # Match original length (n_samples, n_channels, seq_len)

        return x_diff

class RandomCropResize:
    """
    Random-crop-resize augmentation function.

    Parameters
    ----------
    crop_rate: float within [0, 1), default=[0, 0.2]
        How much (in %) of a time series will be cropped.
    size: int, default=None
        To which size the input will be interpolated. By default, to the original sequence length.
    """
    def __init__(self, crop_rate_range=[0, 0.2], size=None):
        self.crop_rate_range = crop_rate_range
        self.size = size
        
    def __call__(self, x):
        """
        Perform augmentation.

        Parameters
        ----------
        x: torch tensor of shape (n_samples, n_channels, seq_len)
            Original time series.
        """
        # sample crop_rate randomly within the range
        crop_rate = torch.empty(1).uniform_(self.crop_rate_range[0], self.crop_rate_range[1]).item()

        # determine the sequence length
        seq_len = x.shape[-1]
        size = seq_len if self.size is None else self.size
        cropped_seq_len = int(seq_len * (1-crop_rate))  # Calculate the length of the cropped sequence

        # generate a random starting index for the crop
        start_idx = torch.randint(0, seq_len - cropped_seq_len + 1, (1,)).item()

        # perform the crop on the time dimension (last dimension)
        x_cropped = x[:, :, start_idx:start_idx+cropped_seq_len]

        # resize the cropped sequence to the target size
        # only need to resize along the time dimension (last dimension)
        x_resized = F.interpolate(x_cropped, size=size, mode='linear', align_corners=False)

        return x_resized


class FastFourierTransform:
    """
    Fast Fourier Transform (FFT) augmentation function.

    Parameters
    ----------
    None
    """
    def __init__(self):
        pass
        
    def __call__(self, x):
        """
        Perform FFT augmentation.

        Parameters
        ----------
        x: torch tensor of shape (n_samples, n_channels, seq_len)
            Original time series.
        """
        # Apply FFT along the time dimension (last dimension)
        x_fft = torch.fft.fft(x, dim=-1)

        # Return the real part of the FFT result
        return x_fft.real