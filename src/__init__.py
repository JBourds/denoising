import librosa
import numpy as np
import soundfile as sf
from matplotlib import pyplot as plt

# Reasonable defaults
N_FFT = 1024
HOP = N_FFT // 4


def plot_td(signal: np.ndarray, denoised: np.ndarray, sr: float, fig_path: str):
    min_len = min(len(signal), len(denoised))
    times = np.arange(len(signal[:min_len])) / sr
    plt.figure(figsize=(14, 5))
    plt.subplot(2, 1, 1)
    plt.title("Original signal signal")
    plt.plot(times, signal[:min_len], linewidth=0.5)
    plt.xlabel("Time (s)")
    plt.subplot(2, 1, 2)
    plt.title("Denoised signal (Spectral Subtraction)")
    plt.plot(times, denoised[:min_len], color="purple", linewidth=0.5)
    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()


def save_audio(y: np.ndarray, sr: float, path: str):
    sf.write(path, y, sr, subtype="PCM_16")


def reconstruct_signal(
    S: np.ndarray,
    hop: int,
    upscale: float,
) -> np.ndarray:
    y_denoised = librosa.istft(S, hop_length=hop)
    y_denoised = np.clip(y_denoised * upscale, -1.0, 1.0)
    return y_denoised
