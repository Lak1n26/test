import random

import torch
import torchaudio
from torch import nn


class GaussianNoise(nn.Module):
    """
    Add Gaussian noise to audio waveform.

    This augmentation simulates recording noise and helps the model
    become more robust to noisy audio inputs.
    """

    def __init__(
        self,
        min_snr_db: float = 10.0,
        max_snr_db: float = 40.0,
        p: float = 0.5,
    ):
        """
        Args:
            min_snr_db (float): Minimum signal-to-noise ratio in dB.
            max_snr_db (float): Maximum signal-to-noise ratio in dB.
            p (float): Probability of applying the augmentation.
        """
        super().__init__()
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db
        self.p = p

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Add Gaussian noise to audio.

        Args:
            audio (Tensor): Audio waveform of shape [time] or [batch, time].
        Returns:
            augmented (Tensor): Audio with added noise.
        """
        if random.random() > self.p:
            return audio

        # Random SNR in dB
        snr_db = random.uniform(self.min_snr_db, self.max_snr_db)

        # Calculate signal power
        signal_power = audio.pow(2).mean()

        # Calculate noise power from SNR
        # SNR = 10 * log10(signal_power / noise_power)
        # noise_power = signal_power / (10 ** (snr_db / 10))
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear

        # Generate noise
        noise = torch.randn_like(audio) * torch.sqrt(noise_power)

        return audio + noise


class Gain(nn.Module):
    """
    Apply random gain (volume change) to audio.

    This augmentation helps the model become invariant to
    different recording volumes.
    """

    def __init__(
        self,
        min_gain_db: float = -12.0,
        max_gain_db: float = 12.0,
        p: float = 0.5,
    ):
        """
        Args:
            min_gain_db (float): Minimum gain in dB.
            max_gain_db (float): Maximum gain in dB.
            p (float): Probability of applying the augmentation.
        """
        super().__init__()
        self.min_gain_db = min_gain_db
        self.max_gain_db = max_gain_db
        self.p = p

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Apply random gain to audio.

        Args:
            audio (Tensor): Audio waveform of shape [time] or [batch, time].
        Returns:
            augmented (Tensor): Audio with gain applied.
        """
        if random.random() > self.p:
            return audio

        gain_db = random.uniform(self.min_gain_db, self.max_gain_db)
        gain_linear = 10 ** (gain_db / 20)  # dB to linear

        return audio * gain_linear


class TimeStretch(nn.Module):
    """
    Apply time stretching to audio spectrogram.

    Changes the speed of audio without changing the pitch.
    Applied on spectrogram level for efficiency.
    """

    def __init__(
        self,
        min_rate: float = 0.8,
        max_rate: float = 1.2,
        n_freq: int = 80,
        hop_length: int = 160,
        p: float = 0.5,
    ):
        """
        Args:
            min_rate (float): Minimum stretch rate (< 1 = slower).
            max_rate (float): Maximum stretch rate (> 1 = faster).
            n_freq (int): Number of frequency bins.
            hop_length (int): Hop length of STFT.
            p (float): Probability of applying the augmentation.
        """
        super().__init__()
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.n_freq = n_freq
        self.hop_length = hop_length
        self.p = p

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply time stretching to spectrogram.

        Args:
            spectrogram (Tensor): Spectrogram [n_mels, time] or [batch, n_mels, time].
        Returns:
            stretched (Tensor): Time-stretched spectrogram.
        """
        if random.random() > self.p:
            return spectrogram

        rate = random.uniform(self.min_rate, self.max_rate)

        # Use interpolation for stretching
        if spectrogram.dim() == 2:
            spectrogram = spectrogram.unsqueeze(0).unsqueeze(0)
            squeeze = True
        else:
            spectrogram = spectrogram.unsqueeze(1)
            squeeze = False

        # New time dimension
        orig_time = spectrogram.shape[-1]
        new_time = int(orig_time / rate)

        # Interpolate
        stretched = torch.nn.functional.interpolate(
            spectrogram, size=(spectrogram.shape[2], new_time), mode="bilinear"
        )

        if squeeze:
            stretched = stretched.squeeze(0).squeeze(0)
        else:
            stretched = stretched.squeeze(1)

        return stretched


class PitchShift(nn.Module):
    """
    Apply pitch shifting to audio.

    Uses torchaudio's pitch shifting functionality.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        min_semitones: float = -4.0,
        max_semitones: float = 4.0,
        p: float = 0.5,
    ):
        """
        Args:
            sample_rate (int): Audio sample rate.
            min_semitones (float): Minimum pitch shift in semitones.
            max_semitones (float): Maximum pitch shift in semitones.
            p (float): Probability of applying the augmentation.
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.min_semitones = min_semitones
        self.max_semitones = max_semitones
        self.p = p

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Apply pitch shifting to audio.

        Args:
            audio (Tensor): Audio waveform of shape [time] or [batch, time].
        Returns:
            shifted (Tensor): Pitch-shifted audio.
        """
        if random.random() > self.p:
            return audio

        n_steps = random.uniform(self.min_semitones, self.max_semitones)

        # Handle batched input
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        # Apply pitch shift
        shifted = torchaudio.functional.pitch_shift(
            audio, self.sample_rate, n_steps
        )

        if squeeze:
            shifted = shifted.squeeze(0)

        return shifted


class SpeedPerturbation(nn.Module):
    """
    Apply speed perturbation to audio.

    Changes the speed (and pitch) of audio by resampling.
    Common augmentation for ASR.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        speeds: list = None,
        p: float = 0.5,
    ):
        """
        Args:
            sample_rate (int): Original sample rate.
            speeds (list): List of speed factors to choose from.
                Default: [0.9, 1.0, 1.1]
            p (float): Probability of applying the augmentation.
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.speeds = speeds if speeds is not None else [0.9, 1.0, 1.1]
        self.p = p

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Apply speed perturbation to audio.

        Args:
            audio (Tensor): Audio waveform of shape [time] or [batch, time].
        Returns:
            perturbed (Tensor): Speed-perturbed audio.
        """
        if random.random() > self.p:
            return audio

        speed = random.choice(self.speeds)

        if speed == 1.0:
            return audio

        # Speed change via resampling
        new_sr = int(self.sample_rate * speed)
        resampler = torchaudio.transforms.Resample(self.sample_rate, new_sr)
        resampler_back = torchaudio.transforms.Resample(new_sr, self.sample_rate)

        # Apply speed change
        perturbed = resampler_back(resampler(audio))

        return perturbed


class AudioAugmentations(nn.Module):
    """
    Compose multiple audio augmentations.

    Applies a sequence of audio augmentations in order.
    """

    def __init__(
        self,
        augmentations: list = None,
        sample_rate: int = 16000,
    ):
        """
        Args:
            augmentations (list): List of augmentation configs.
                If None, uses default set of augmentations.
            sample_rate (int): Audio sample rate.
        """
        super().__init__()

        if augmentations is None:
            self.augmentations = nn.ModuleList([
                GaussianNoise(min_snr_db=15, max_snr_db=35, p=0.3),
                Gain(min_gain_db=-6, max_gain_db=6, p=0.3),
                SpeedPerturbation(sample_rate=sample_rate, p=0.3),
            ])
        else:
            self.augmentations = nn.ModuleList(augmentations)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Apply all augmentations in sequence.

        Args:
            audio (Tensor): Audio waveform.
        Returns:
            augmented (Tensor): Augmented audio.
        """
        for aug in self.augmentations:
            audio = aug(audio)
        return audio

