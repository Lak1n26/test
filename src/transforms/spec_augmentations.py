import random

import torch
from torch import nn


class FrequencyMasking(nn.Module):
    """
    Apply frequency masking to spectrogram (SpecAugment).

    Masks a random contiguous band of frequency bins.
    Paper: https://arxiv.org/abs/1904.08779
    """

    def __init__(
        self,
        freq_mask_param: int = 27,
        num_masks: int = 1,
        p: float = 1.0,
        mask_value: float = 0.0,
    ):
        """
        Args:
            freq_mask_param (int): Maximum width of frequency mask (F parameter).
            num_masks (int): Number of frequency masks to apply.
            p (float): Probability of applying the augmentation.
            mask_value (float): Value to fill masked regions.
        """
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.num_masks = num_masks
        self.p = p
        self.mask_value = mask_value

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply frequency masking to spectrogram.

        Args:
            spectrogram (Tensor): Spectrogram of shape [n_mels, time]
                or [batch, n_mels, time].
        Returns:
            masked (Tensor): Frequency-masked spectrogram.
        """
        if random.random() > self.p:
            return spectrogram

        # Handle batched input
        if spectrogram.dim() == 2:
            spectrogram = spectrogram.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        batch_size, n_freq, n_time = spectrogram.shape
        spectrogram = spectrogram.clone()

        for _ in range(self.num_masks):
            # Random mask width
            f = random.randint(0, min(self.freq_mask_param, n_freq))
            if f == 0:
                continue

            # Random starting frequency
            f0 = random.randint(0, n_freq - f)

            # Apply mask
            spectrogram[:, f0 : f0 + f, :] = self.mask_value

        if squeeze:
            spectrogram = spectrogram.squeeze(0)

        return spectrogram


class TimeMasking(nn.Module):
    """
    Apply time masking to spectrogram (SpecAugment).

    Masks a random contiguous segment of time frames.
    Paper: https://arxiv.org/abs/1904.08779
    """

    def __init__(
        self,
        time_mask_param: int = 100,
        num_masks: int = 1,
        p: float = 1.0,
        mask_value: float = 0.0,
        max_mask_ratio: float = 1.0,
    ):
        """
        Args:
            time_mask_param (int): Maximum width of time mask (T parameter).
            num_masks (int): Number of time masks to apply.
            p (float): Probability of applying the augmentation.
            mask_value (float): Value to fill masked regions.
            max_mask_ratio (float): Maximum ratio of time to mask
                (to prevent masking entire spectrogram).
        """
        super().__init__()
        self.time_mask_param = time_mask_param
        self.num_masks = num_masks
        self.p = p
        self.mask_value = mask_value
        self.max_mask_ratio = max_mask_ratio

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply time masking to spectrogram.

        Args:
            spectrogram (Tensor): Spectrogram of shape [n_mels, time]
                or [batch, n_mels, time].
        Returns:
            masked (Tensor): Time-masked spectrogram.
        """
        if random.random() > self.p:
            return spectrogram

        # Handle batched input
        if spectrogram.dim() == 2:
            spectrogram = spectrogram.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        batch_size, n_freq, n_time = spectrogram.shape
        spectrogram = spectrogram.clone()

        # Maximum mask width based on max_mask_ratio
        max_t = min(self.time_mask_param, int(n_time * self.max_mask_ratio))

        for _ in range(self.num_masks):
            # Random mask width
            t = random.randint(0, max_t)
            if t == 0:
                continue

            # Random starting time
            t0 = random.randint(0, max(0, n_time - t))

            # Apply mask
            spectrogram[:, :, t0 : t0 + t] = self.mask_value

        if squeeze:
            spectrogram = spectrogram.squeeze(0)

        return spectrogram


class SpecAugment(nn.Module):
    """
    SpecAugment: Apply both frequency and time masking.

    Combines frequency and time masking as described in the
    SpecAugment paper.

    Paper: https://arxiv.org/abs/1904.08779
    """

    def __init__(
        self,
        freq_mask_param: int = 27,
        time_mask_param: int = 100,
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
        mask_value: float = 0.0,
        max_time_mask_ratio: float = 1.0,
        p: float = 1.0,
    ):
        """
        Args:
            freq_mask_param (int): Maximum width of frequency mask.
            time_mask_param (int): Maximum width of time mask.
            num_freq_masks (int): Number of frequency masks.
            num_time_masks (int): Number of time masks.
            mask_value (float): Value to fill masked regions.
            max_time_mask_ratio (float): Maximum ratio of time to mask.
            p (float): Probability of applying the augmentation.
        """
        super().__init__()
        self.p = p
        self.freq_mask = FrequencyMasking(
            freq_mask_param=freq_mask_param,
            num_masks=num_freq_masks,
            p=1.0,  # Always apply when SpecAugment is applied
            mask_value=mask_value,
        )
        self.time_mask = TimeMasking(
            time_mask_param=time_mask_param,
            num_masks=num_time_masks,
            p=1.0,  # Always apply when SpecAugment is applied
            mask_value=mask_value,
            max_mask_ratio=max_time_mask_ratio,
        )

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment (frequency + time masking).

        Args:
            spectrogram (Tensor): Spectrogram of shape [n_mels, time]
                or [batch, n_mels, time].
        Returns:
            augmented (Tensor): Augmented spectrogram.
        """
        if random.random() > self.p:
            return spectrogram

        spectrogram = self.freq_mask(spectrogram)
        spectrogram = self.time_mask(spectrogram)

        return spectrogram


class TimeWarp(nn.Module):
    """
    Apply time warping to spectrogram (SpecAugment).

    Warps the spectrogram along the time axis using interpolation.
    This is the 'W' parameter in SpecAugment.
    """

    def __init__(
        self,
        warp_param: int = 80,
        p: float = 0.5,
    ):
        """
        Args:
            warp_param (int): Time warp parameter (W).
            p (float): Probability of applying the augmentation.
        """
        super().__init__()
        self.warp_param = warp_param
        self.p = p

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply time warping to spectrogram.

        Args:
            spectrogram (Tensor): Spectrogram of shape [n_mels, time]
                or [batch, n_mels, time].
        Returns:
            warped (Tensor): Time-warped spectrogram.
        """
        if random.random() > self.p:
            return spectrogram

        # Handle batched input
        if spectrogram.dim() == 2:
            spectrogram = spectrogram.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        batch_size, n_freq, n_time = spectrogram.shape

        if n_time <= 2 * self.warp_param:
            if squeeze:
                return spectrogram.squeeze(0)
            return spectrogram

        # Random source and destination points
        W = self.warp_param
        source_pt = random.randint(W, n_time - W)
        dest_pt = source_pt + random.randint(-W, W)

        # Create warping function
        # Linear interpolation between source and dest
        # Create new time indices
        left_ratio = dest_pt / source_pt
        right_ratio = (n_time - dest_pt) / (n_time - source_pt)

        indices = torch.zeros(n_time, device=spectrogram.device)
        for t in range(n_time):
            if t < dest_pt:
                indices[t] = t / left_ratio
            else:
                indices[t] = source_pt + (t - dest_pt) / right_ratio

        # Clamp indices
        indices = torch.clamp(indices, 0, n_time - 1)

        # Interpolate
        # Use grid_sample for proper interpolation
        spectrogram_4d = spectrogram.unsqueeze(1)  # [B, 1, F, T]

        # Create grid
        grid_y = torch.linspace(-1, 1, n_freq, device=spectrogram.device)
        grid_x = (indices / (n_time - 1)) * 2 - 1  # Normalize to [-1, 1]

        grid = torch.stack(
            torch.meshgrid(grid_y, grid_x, indexing="ij"), dim=-1
        )  # [F, T, 2]
        grid = grid.unsqueeze(0).expand(batch_size, -1, -1, -1)  # [B, F, T, 2]

        # Apply grid sample
        warped = torch.nn.functional.grid_sample(
            spectrogram_4d,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

        warped = warped.squeeze(1)  # [B, F, T]

        if squeeze:
            warped = warped.squeeze(0)

        return warped

