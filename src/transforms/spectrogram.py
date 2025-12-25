import torch
import torchaudio
from torch import nn


class MelSpectrogram(nn.Module):
    """
    Compute mel spectrogram from audio waveform.

    This transform converts raw audio waveforms into mel spectrograms,
    which are commonly used as input features for ASR models.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        win_length: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: float = 8000.0,
        power: float = 2.0,
        pad_value: float = -11.52,  # log(1e-5)
    ):
        """
        Args:
            sample_rate (int): Sample rate of audio signal.
            n_fft (int): Size of FFT.
            win_length (int): Window size.
            hop_length (int): Length of hop between STFT windows.
            n_mels (int): Number of mel filterbanks.
            f_min (float): Minimum frequency.
            f_max (float): Maximum frequency.
            power (float): Exponent for the magnitude spectrogram.
            pad_value (float): Value for padding in log mel spectrogram.
        """
        super().__init__()

        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.pad_value = pad_value

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=power,
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Compute mel spectrogram from audio waveform.

        Args:
            audio (Tensor): Audio waveform of shape [time] or [batch, time].
        Returns:
            mel_spec (Tensor): Log mel spectrogram of shape [n_mels, time]
                or [batch, n_mels, time].
        """
        mel_spec = self.mel_spectrogram(audio)

        # Convert to log scale
        mel_spec = torch.log(mel_spec.clamp(min=1e-5))

        return mel_spec


class MelSpectrogramWithResample(nn.Module):
    """
    Compute mel spectrogram with optional resampling.

    Handles audio with different sample rates by resampling
    to the target sample rate before computing mel spectrogram.
    """

    def __init__(
        self,
        target_sample_rate: int = 16000,
        n_fft: int = 400,
        win_length: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: float = 8000.0,
        power: float = 2.0,
    ):
        """
        Args:
            target_sample_rate (int): Target sample rate for resampling.
            n_fft (int): Size of FFT.
            win_length (int): Window size.
            hop_length (int): Length of hop between STFT windows.
            n_mels (int): Number of mel filterbanks.
            f_min (float): Minimum frequency.
            f_max (float): Maximum frequency.
            power (float): Exponent for the magnitude spectrogram.
        """
        super().__init__()

        self.target_sample_rate = target_sample_rate
        self._resamplers = {}  # Cache for resamplers with different source rates

        self.mel_spectrogram = MelSpectrogram(
            sample_rate=target_sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=power,
        )

    def _get_resampler(self, source_sr: int) -> torchaudio.transforms.Resample:
        """Get or create a resampler for the given source sample rate."""
        if source_sr not in self._resamplers:
            self._resamplers[source_sr] = torchaudio.transforms.Resample(
                orig_freq=source_sr,
                new_freq=self.target_sample_rate,
            )
        return self._resamplers[source_sr]

    def forward(
        self, audio: torch.Tensor, sample_rate: int = None
    ) -> torch.Tensor:
        """
        Compute mel spectrogram, resampling if necessary.

        Args:
            audio (Tensor): Audio waveform of shape [time] or [batch, time].
            sample_rate (int): Source sample rate. If different from target,
                audio will be resampled.
        Returns:
            mel_spec (Tensor): Log mel spectrogram.
        """
        # Resample if needed
        if sample_rate is not None and sample_rate != self.target_sample_rate:
            resampler = self._get_resampler(sample_rate)
            resampler = resampler.to(audio.device)
            audio = resampler(audio)

        return self.mel_spectrogram(audio)


class SpectrogramNormalize(nn.Module):
    """
    Normalize spectrogram features.

    Can perform per-instance normalization (zero mean, unit variance)
    or use pre-computed global statistics.
    """

    def __init__(
        self,
        mean: float = None,
        std: float = None,
        per_instance: bool = True,
    ):
        """
        Args:
            mean (float): Global mean for normalization. If None and
                per_instance=False, no mean subtraction is performed.
            std (float): Global std for normalization. If None and
                per_instance=False, no std division is performed.
            per_instance (bool): If True, normalize each spectrogram
                individually using its own statistics.
        """
        super().__init__()

        self.per_instance = per_instance
        self.register_buffer(
            "mean", torch.tensor(mean) if mean is not None else None
        )
        self.register_buffer(
            "std", torch.tensor(std) if std is not None else None
        )

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Normalize spectrogram.

        Args:
            spectrogram (Tensor): Spectrogram of shape [n_mels, time]
                or [batch, n_mels, time].
        Returns:
            normalized (Tensor): Normalized spectrogram.
        """
        if self.per_instance:
            # Compute statistics over frequency and time dimensions
            mean = spectrogram.mean()
            std = spectrogram.std()
            return (spectrogram - mean) / (std + 1e-5)
        else:
            result = spectrogram
            if self.mean is not None:
                result = result - self.mean
            if self.std is not None:
                result = result / (self.std + 1e-5)
            return result

