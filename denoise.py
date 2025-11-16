import argparse
from functools import partial
from typing import Callable, Optional

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import wiener

# Reasonable defaults
N_FFT = 1024
HOP = N_FFT // 4


def spectral_subtraction(
    noise: np.ndarray,
    signal: np.ndarray,
    alpha: float,
    n_fft: int = N_FFT,
    hop: int = HOP,
) -> np.ndarray:
    N = librosa.stft(noise, n_fft=n_fft, hop_length=hop)
    Y = librosa.stft(signal, n_fft=n_fft, hop_length=hop)
    noise_mag = np.mean(np.abs(N), axis=1, keepdims=True)
    Y_mag = np.abs(Y)
    Y_phase = np.angle(Y)
    S_mag = np.maximum(Y_mag - alpha * noise_mag, 0.0)
    S = S_mag * np.exp(1j * Y_phase)
    return S


def amplitude_threshold(
    signal: np.ndarray,
    value: Optional[float] = None,
    percent: Optional[float] = None,
    n_fft: int = N_FFT,
    hop: int = HOP,
) -> np.ndarray:
    if (value is None) == (percent is None):
        raise ValueError("Must provide a single fixed threshold value or percentage.")
    y = librosa.stft(signal, n_fft=n_fft, hop_length=hop)
    y_mag = np.abs(y)
    threshold = value if value is not None else y_mag.max() * percent
    mask = y_mag < threshold
    y[mask] = 0
    return y


def reconstruct_signal(
    S: np.ndarray,
    hop: int,
    upscale: float,
) -> np.ndarray:
    y_denoised = librosa.istft(S, hop_length=hop)
    y_denoised = np.clip(y_denoised * upscale, -1.0, 1.0)
    return y_denoised


def filter_signal(y: np.ndarray, iterations: int) -> np.ndarray:
    for _ in range(iterations):
        y = wiener(y)
    return y


def save_audio(y: np.ndarray, sr: float, path: str):
    sf.write(path, y, sr, subtype="PCM_16")


def plot_signals(signal: np.ndarray, denoised: np.ndarray, fig_path: str):
    plt.figure(figsize=(14, 5))
    plt.subplot(2, 1, 1)
    plt.title("Original signal signal")
    plt.plot(signal, linewidth=0.5)
    plt.subplot(2, 1, 2)
    plt.title("Denoised signal (Spectral Subtraction)")
    plt.plot(denoised, color="purple", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Spectral subtraction denoising script."
    )
    parser.add_argument("--noise", help="Path to noise-only WAV file.")
    parser.add_argument(
        "--signal", required=True, help="Path to signal + noise WAV file."
    )
    parser.add_argument("--out", required=True, help="Path to save denoised WAV file.")
    parser.add_argument(
        "--alpha", type=float, default=1.0, help="Spectral subtraction scaling factor."
    )
    parser.add_argument(
        "--amp_value",
        type=float,
        help="Fixed amplitude value to set signal frequency content to 0 beneath.",
    )
    parser.add_argument(
        "--amp_percent",
        type=float,
        help="Percentage of maximum amplitude below which to set to 0.",
    )
    parser.add_argument(
        "--upscale", type=float, default=1.0, help="Output amplitude scaling."
    )
    parser.add_argument(
        "--plot",
        default="denoised.png",
        help="Filename to save comparison plot.",
    )
    parser.add_argument(
        "--filters",
        type=int,
        default=0,
        help="Numbe of rounds of wiener filtering to apply.",
    )
    args = parser.parse_args()
    signal, sr = librosa.load(args.signal, sr=None, mono=True)
    if args.noise is not None:
        noise, _ = librosa.load(args.noise, sr=None, mono=True)
        S = spectral_subtraction(noise, signal, args.alpha)
        y_denoised = reconstruct_signal(S, HOP, args.upscale)
    else:
        S = amplitude_threshold(signal, args.amp_value, args.amp_percent)
        y_denoised = reconstruct_signal(S, HOP, args.upscale)

    for _ in range(args.filters):
        y_denoised = wiener(y_denoised)

    save_audio(y_denoised, sr, args.out)
    plot_signals(signal, y_denoised, args.plot)
    print(f"Saved denoised file: {args.out}")
    print(f"Saved plot: {args.plot}")


if __name__ == "__main__":
    main()
