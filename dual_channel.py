import argparse

import librosa
import numpy as np
from scipy.signal import wiener

from src import HOP, N_FFT, plot_td, reconstruct_signal, save_audio


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


def filter_signal(y: np.ndarray, iterations: int) -> np.ndarray:
    for _ in range(iterations):
        y = wiener(y)
    return y


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
        help="Number of rounds of wiener filtering to apply.",
    )
    args = parser.parse_args()
    signal, sr = librosa.load(args.signal, sr=None, mono=True)
    noise, _ = librosa.load(args.noise, sr=None, mono=True)
    S = spectral_subtraction(noise, signal, args.alpha)
    y_denoised = args.upscale * reconstruct_signal(S, HOP, args.upscale)

    for _ in range(args.filters):
        y_denoised = wiener(y_denoised)

    save_audio(y_denoised, sr, args.out)
    plot_td(signal, y_denoised, sr, args.plot)
    print(f"Saved denoised file: {args.out}")
    print(f"Saved plot: {args.plot}")


if __name__ == "__main__":
    main()
