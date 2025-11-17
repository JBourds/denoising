import argparse
from typing import Optional

import librosa
import numpy as np
import pywt
from scipy import signal
from scipy.ndimage import median_filter
from scipy.signal import butter, wiener

from src import HOP, N_FFT, plot_td, reconstruct_signal, save_audio


def amplitude_threshold(
    signal: np.ndarray,
    value: Optional[float] = None,
    percent: Optional[float] = None,
    n_fft: int = N_FFT,
    hop: int = HOP,
) -> np.ndarray:
    if (value is None) == (percent is None):
        raise ValueError("Must provide a single fixed threshold value OR percentage.")
    y = librosa.stft(signal, n_fft=n_fft, hop_length=hop)
    y_mag = np.abs(y)
    threshold = value if value is not None else y_mag.max() * percent
    mask = y_mag < threshold
    y[mask] = 0
    return y


def lpf(data: np.ndarray, order: int, sr: int, cutoff_freq: float):
    nyquist = 0.5 * sr
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return signal.filtfilt(b, a, data)


def adaptive_hpf(x: np.ndarray, sr: int, cutoff: int = 400):
    sos = signal.butter(4, cutoff / (sr / 2), "highpass", output="sos")
    return signal.sosfiltfilt(sos, x)


def remove_background_median(x: np.ndarray, kernel: int = 501):
    smooth = median_filter(x, size=kernel)
    return x - smooth


def transient_teager_kaiser(x: np.ndarray):
    x = np.asarray(x)
    y = np.zeros_like(x)
    y[1:-1] = x[1:-1] ** 2 - x[0:-2] * x[2:]
    return y


def transient_teager_smooth(x: np.ndarray, smooth=7):
    raw = transient_teager_kaiser(x)
    kernel = np.ones(smooth) / smooth
    return np.convolve(raw, kernel, mode="same")


def wavelet_denoise(x: np.ndarray, wavelet="sym8", level=4):
    coeffs = pywt.wavedec(x, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthr = sigma * np.sqrt(2 * np.log(len(x)))
    coeffs_filt = [pywt.threshold(c, uthr, mode="soft") for c in coeffs]
    return pywt.waverec(coeffs_filt, wavelet)


def main():
    parser = argparse.ArgumentParser(description="Single channel denoising script.")

    parser.add_argument("--signal", required=True, help="Path to WAV file with signal.")
    parser.add_argument("--out", default="denoised.wav", help="Output WAV file.")
    parser.add_argument("--plot", default="denoised.png", help="Output plot file.")

    parser.add_argument(
        "--filter",
        type=str,
        required=True,
        choices=[
            "wiener",
            "lp",
            "hp",
            "median",
            "transient",
            "wavelet",
            "adaptive_hpf",
            "amp_thresh",
        ],
        help="Denoising filter to apply.",
    )

    # Common options
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Iterations for filters that support it (e.g., wiener).",
    )
    parser.add_argument("--cutoff", type=float, help="Cutoff freq for HP/LP filters.")

    # Median background subtraction
    parser.add_argument(
        "--kernel", type=int, default=512, help="Median filter kernel size."
    )

    # Wavelet denoising options
    parser.add_argument(
        "--wavelet", type=str, default="sym8", help="Wavelet for wavelet denoising."
    )
    parser.add_argument(
        "--upscale", type=float, default=1.0, help="Output amplitude scaling."
    )
    parser.add_argument(
        "--level",
        type=int,
        default=4,
        help="Decomposition level for wavelet denoising.",
    )

    # Amplitude threshold options
    parser.add_argument(
        "--amp-value",
        type=float,
        default=None,
        help="Absolute amplitude threshold value.",
    )
    parser.add_argument(
        "--amp-percent",
        type=float,
        default=None,
        help="Percent of max amplitude to threshold (0–1).",
    )

    args = parser.parse_args()

    data, sr = librosa.load(args.signal, sr=None, mono=True)

    # Select filter
    match args.filter.lower():
        case "wiener":
            y_denoised = data
            for _ in range(args.iterations):
                y_denoised = wiener(y_denoised)
        case "lp":
            if args.cutoff is None:
                raise ValueError("--cutoff required for lp filter")
            y_denoised = lpf(data, order=4, sr=sr, cutoff_freq=args.cutoff)
        case "hp":
            if args.cutoff is None:
                raise ValueError("--cutoff required for hp filter")
            y_denoised = adaptive_hpf(data, sr, cutoff=args.cutoff)
        case "median":
            y_denoised = remove_background_median(data, kernel=args.kernel)
        case "transient":
            y_denoised = transient_teager_smooth(data)
        case "wavelet":
            y_denoised = wavelet_denoise(data, wavelet=args.wavelet, level=args.level)
        case "adaptive_hpf":
            if args.cutoff is None:
                raise ValueError("--cutoff required for adaptive_hpf")
            y_denoised = adaptive_hpf(data, sr, cutoff=args.cutoff)
        case "amp_thresh":
            y_stft = amplitude_threshold(
                data,
                value=args.amp_value,
                percent=args.amp_percent,
            )
            y_denoised = reconstruct_signal(y_stft, HOP, 1.0)
        case _:
            raise ValueError(f"Unknown filter type: {args.filter}")

    y_denoised = args.upscale * y_denoised
    save_audio(y_denoised, sr, args.out)
    plot_td(data, y_denoised, sr, args.plot)
    print(f"Saved denoised file: {args.out}")
    print(f"Saved plot: {args.plot}")


if __name__ == "__main__":
    main()
