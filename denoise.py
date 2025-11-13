import argparse

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import wiener


def load_audio(
    noise_path: str, signal_path: str
) -> tuple[np.ndarray, np.ndarray, float]:
    noise, sr1 = librosa.load(noise_path, sr=None, mono=True)
    signal, sr2 = librosa.load(signal_path, sr=None, mono=True)
    if sr1 != sr2:
        raise ValueError("Sample rates must match.")
    return noise, signal, sr1


def spectral_subtraction(
    noise: np.ndarray, signal: np.ndarray, alpha: float
) -> tuple[np.ndarray, int]:
    n_fft = 1024
    hop = n_fft // 4
    N = librosa.stft(noise, n_fft=n_fft, hop_length=hop)
    Y = librosa.stft(signal, n_fft=n_fft, hop_length=hop)
    noise_mag = np.mean(np.abs(N), axis=1, keepdims=True)
    Y_mag = np.abs(Y)
    Y_phase = np.angle(Y)
    S_mag = np.maximum(Y_mag - alpha * noise_mag, 0.0)
    S = S_mag * np.exp(1j * Y_phase)
    return S, hop


def reconstruct_signal(
    S: np.ndarray, hop: int, upscale: float, iterations: int
) -> np.ndarray:
    y_denoised = librosa.istft(S, hop_length=hop)
    y_denoised = np.clip(y_denoised * upscale, -1.0, 1.0)
    for _ in range(iterations):
        y_denoised = wiener(y_denoised)
    return y_denoised


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
    parser.add_argument("--noise", required=True, help="Path to noise-only WAV file.")
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
    args = parser.parse_args()

    noise, signal, sr = load_audio(args.noise, args.signal)
    S, hop = spectral_subtraction(noise, signal, args.alpha)
    y_denoised = reconstruct_signal(S, hop, args.upscale, 500)
    save_audio(y_denoised, sr, args.out)
    plot_signals(signal, y_denoised, args.plot)
    print(f"Saved denoised file: {args.out}")
    print(f"Saved plot: {args.plot}")


if __name__ == "__main__":
    main()
