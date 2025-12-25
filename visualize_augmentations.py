#!/usr/bin/env python3
"""
Visualize audio and spectrogram augmentations.

Usage:
    python visualize_augmentations.py --audio_path path/to/audio.wav
    python visualize_augmentations.py  # Uses sample from LibriSpeech
"""
import argparse
import os

import matplotlib.pyplot as plt
import soundfile as sf
import torch

from src.transforms import MelSpectrogram
from src.transforms.audio_augmentations import (
    GaussianNoise,
    Gain,
    PitchShift,
    SpeedPerturbation,
)
from src.transforms.spec_augmentations import (
    FrequencyMasking,
    TimeMasking,
    SpecAugment,
    TimeWarp,
)


def load_audio(path: str) -> tuple:
    """Load audio file."""
    audio_np, sr = sf.read(path)
    audio = torch.from_numpy(audio_np).float()
    if audio.ndim > 1:
        audio = audio.mean(dim=-1)
    return audio, sr


def plot_waveforms(original, augmented, title, sr=16000):
    """Plot original and augmented waveforms."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 4))
    
    time_orig = torch.arange(len(original)) / sr
    time_aug = torch.arange(len(augmented)) / sr
    
    axes[0].plot(time_orig, original.numpy(), alpha=0.7)
    axes[0].set_title("Original Audio")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    
    axes[1].plot(time_aug, augmented.numpy(), alpha=0.7, color='orange')
    axes[1].set_title(f"After {title}")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Amplitude")
    
    plt.tight_layout()
    return fig


def plot_spectrograms(original, augmented, title):
    """Plot original and augmented spectrograms."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    
    im0 = axes[0].imshow(original.numpy(), aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title("Original Spectrogram")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Mel Bin")
    plt.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].imshow(augmented.numpy(), aspect='auto', origin='lower', cmap='viridis')
    axes[1].set_title(f"After {title}")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Mel Bin")
    plt.colorbar(im1, ax=axes[1])
    
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_path", type=str, default=None,
                       help="Path to audio file. If not provided, uses sample from LibriSpeech.")
    parser.add_argument("--output_dir", type=str, default="augmentation_examples",
                       help="Directory to save visualizations.")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find sample audio
    if args.audio_path:
        audio_path = args.audio_path
    else:
        # Try to find a sample from LibriSpeech
        libri_path = "data/datasets/librispeech/LibriSpeech"
        if os.path.exists(libri_path):
            for root, dirs, files in os.walk(libri_path):
                for f in files:
                    if f.endswith('.flac'):
                        audio_path = os.path.join(root, f)
                        break
                if 'audio_path' in dir():
                    break
        else:
            print("No audio file found. Please provide --audio_path")
            return
    
    print(f"Using audio: {audio_path}")
    
    # Load audio
    audio, sr = load_audio(audio_path)
    print(f"Audio shape: {audio.shape}, Sample rate: {sr}")
    
    # Initialize mel spectrogram transform
    mel_spec = MelSpectrogram(sample_rate=16000)
    
    # Compute original spectrogram
    spec_original = mel_spec(audio)
    print(f"Spectrogram shape: {spec_original.shape}")
    
    # ========== AUDIO AUGMENTATIONS ==========
    print("\n=== Audio Augmentations ===")
    
    # 1. Gaussian Noise
    print("1. Gaussian Noise")
    noise_aug = GaussianNoise(min_snr_db=10, max_snr_db=20, p=1.0)
    audio_noise = noise_aug(audio.clone())
    fig = plot_waveforms(audio, audio_noise, "Gaussian Noise", sr)
    fig.savefig(f"{args.output_dir}/1_gaussian_noise_waveform.png", dpi=150)
    plt.close()
    
    spec_noise = mel_spec(audio_noise)
    fig = plot_spectrograms(spec_original, spec_noise, "Gaussian Noise")
    fig.savefig(f"{args.output_dir}/1_gaussian_noise_spectrogram.png", dpi=150)
    plt.close()
    
    # 2. Gain
    print("2. Gain (Volume Change)")
    gain_aug = Gain(min_gain_db=-10, max_gain_db=10, p=1.0)
    audio_gain = gain_aug(audio.clone())
    fig = plot_waveforms(audio, audio_gain, "Gain", sr)
    fig.savefig(f"{args.output_dir}/2_gain_waveform.png", dpi=150)
    plt.close()
    
    spec_gain = mel_spec(audio_gain)
    fig = plot_spectrograms(spec_original, spec_gain, "Gain")
    fig.savefig(f"{args.output_dir}/2_gain_spectrogram.png", dpi=150)
    plt.close()
    
    # 3. Speed Perturbation
    print("3. Speed Perturbation")
    speed_aug = SpeedPerturbation(sample_rate=sr, speeds=[0.9], p=1.0)
    audio_speed = speed_aug(audio.clone())
    fig = plot_waveforms(audio, audio_speed, "Speed Perturbation (0.9x)", sr)
    fig.savefig(f"{args.output_dir}/3_speed_perturbation_waveform.png", dpi=150)
    plt.close()
    
    spec_speed = mel_spec(audio_speed)
    fig = plot_spectrograms(spec_original, spec_speed, "Speed Perturbation (0.9x)")
    fig.savefig(f"{args.output_dir}/3_speed_perturbation_spectrogram.png", dpi=150)
    plt.close()
    
    # 4. Pitch Shift
    print("4. Pitch Shift")
    pitch_aug = PitchShift(sample_rate=sr, min_semitones=3, max_semitones=3, p=1.0)
    audio_pitch = pitch_aug(audio.clone())
    fig = plot_waveforms(audio, audio_pitch, "Pitch Shift (+3 semitones)", sr)
    fig.savefig(f"{args.output_dir}/4_pitch_shift_waveform.png", dpi=150)
    plt.close()
    
    spec_pitch = mel_spec(audio_pitch)
    fig = plot_spectrograms(spec_original, spec_pitch, "Pitch Shift (+3 semitones)")
    fig.savefig(f"{args.output_dir}/4_pitch_shift_spectrogram.png", dpi=150)
    plt.close()
    
    # ========== SPECTROGRAM AUGMENTATIONS ==========
    print("\n=== Spectrogram Augmentations ===")
    
    # 5. Frequency Masking
    print("5. Frequency Masking")
    freq_mask = FrequencyMasking(freq_mask_param=27, num_masks=2, p=1.0)
    spec_freq_masked = freq_mask(spec_original.clone())
    fig = plot_spectrograms(spec_original, spec_freq_masked, "Frequency Masking")
    fig.savefig(f"{args.output_dir}/5_frequency_masking.png", dpi=150)
    plt.close()
    
    # 6. Time Masking
    print("6. Time Masking")
    time_mask = TimeMasking(time_mask_param=50, num_masks=2, p=1.0)
    spec_time_masked = time_mask(spec_original.clone())
    fig = plot_spectrograms(spec_original, spec_time_masked, "Time Masking")
    fig.savefig(f"{args.output_dir}/6_time_masking.png", dpi=150)
    plt.close()
    
    # 7. SpecAugment (Combined)
    print("7. SpecAugment (Freq + Time Masking)")
    spec_aug = SpecAugment(
        freq_mask_param=27,
        time_mask_param=50,
        num_freq_masks=2,
        num_time_masks=2,
        p=1.0
    )
    spec_augmented = spec_aug(spec_original.clone())
    fig = plot_spectrograms(spec_original, spec_augmented, "SpecAugment")
    fig.savefig(f"{args.output_dir}/7_specaugment.png", dpi=150)
    plt.close()
    
    # 8. Time Warp
    print("8. Time Warp")
    time_warp = TimeWarp(warp_param=40, p=1.0)
    spec_warped = time_warp(spec_original.clone())
    fig = plot_spectrograms(spec_original, spec_warped, "Time Warp")
    fig.savefig(f"{args.output_dir}/8_time_warp.png", dpi=150)
    plt.close()
    
    print(f"\n✓ All visualizations saved to: {args.output_dir}/")
    print("\nAugmentations implemented:")
    print("  Audio: GaussianNoise, Gain, SpeedPerturbation, PitchShift")
    print("  Spectrogram: FrequencyMasking, TimeMasking, SpecAugment, TimeWarp")


if __name__ == "__main__":
    main()

