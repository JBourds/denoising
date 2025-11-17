import argparse
import os
from typing import Optional

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

ROOT = os.path.dirname(os.path.realpath(__file__))
DB = "noisedb"
NOISES = [
    "-".join(file.strip(".wav").split()) for file in os.listdir(os.path.join(ROOT, DB))
]


def sample_wav(n: int, filepath: str) -> np.ndarray:
    audio, _ = librosa.load(filepath, sr=None, mono=True)
    y = audio.copy()
    while len(audio) < n:
        y = np.concatenate((y, audio), axis=0)
    return y[:n]


def sine_noise(
    length: float, sr: int, freq: float = 100, amp: float = 0.1
) -> np.ndarray:
    t = np.linspace(0, length, int(sr * length), endpoint=False)
    return amp * np.sin(2 * np.pi * freq * t)


def multi_sine_noise(
    length: float, sr: int, freqs: list[float], amps: list[float]
) -> np.ndarray:
    t = np.linspace(0, length, int(sr * length), endpoint=False)
    sig = np.zeros_like(t)
    for f, a in zip(freqs, amps):
        sig += a * np.sin(2 * np.pi * f * t)
    return sig


def pink_noise(n: int) -> np.ndarray:
    num_rows = 16
    array = np.random.randn(num_rows, n)
    array = np.cumsum(array, axis=1)
    weights = 2 ** np.arange(num_rows)
    return np.dot(weights, array) / weights.sum()


def wind_noise(
    length: float, sr: int, amp: float = 0.2, cutoff: float = 1000.0
) -> np.ndarray:
    n = int(sr * length)
    white = np.random.randn(n)
    sos = butter(4, cutoff / (sr / 2), btype="low", output="sos")
    wind = sosfilt(sos, white)
    wind = wind / np.max(np.abs(wind))
    return amp * wind


def apply_gusts(
    wind: np.ndarray,
    sr: int,
    gust_depth: float = 0.7,
    min_rate: float = 0.05,
    max_rate: float = 1.0,
    segment_ratio: float = 0.01,
) -> np.ndarray:
    n = len(wind)

    # Generate slow random variations
    random_noise = np.random.randn(n)

    # Choose a random cutoff for the lowpass filter for each gust segment
    segment_len = int(n * segment_ratio)
    envelope = np.zeros(n)

    for start in range(0, n, segment_len):
        end = min(start + segment_len, n)
        cutoff = np.random.uniform(min_rate, max_rate) / (sr / 2)
        sos = butter(2, cutoff, btype="low", output="sos")
        envelope[start:end] = sosfilt(sos, random_noise[start:end])

    # Normalize envelope to [1, 1+gust_depth]
    envelope -= envelope.min()
    envelope /= envelope.max()
    envelope = 1 + gust_depth * envelope

    return wind * envelope


def ambulance_siren(
    length: float,
    sr: int,
    low: float = 550,
    high: float = 800,
    rate: float = 1.4,
    amp: float = 0.2,
) -> np.ndarray:
    t = np.linspace(0, length, int(sr * length), endpoint=False)
    mod = (np.sin(2 * np.pi * rate * t) + 1) / 2
    freq = low + (high - low) * mod
    phase = np.cumsum(freq) / sr
    siren = np.sin(2 * np.pi * phase)
    return amp * siren


def gaussian_noise(length: float, sr: int, amp: float = 0.1) -> np.ndarray:
    return amp * np.random.randn(int(sr * length))


def rms(x: np.ndarray) -> float:
    return np.sqrt(np.mean(x**2))


def mix(
    signal: np.ndarray, noise: np.ndarray, snr_db: float = 20.0
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    signal_rms = rms(signal)
    noise_rms = rms(noise)
    if noise_rms == 0:
        return signal, None
    target_noise_rms = signal_rms / (10 ** (snr_db / 20))
    scaled_noise = noise * (target_noise_rms / noise_rms)
    out = signal + scaled_noise
    max_val = np.max(np.abs(out))
    if max_val > 1:
        out = out / max_val
        scaled_noise = scaled_noise / max_val
    return out, scaled_noise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inject various noise types into a WAV file."
    )
    parser.add_argument("--input", required=True, help="Input WAV file.")
    parser.add_argument("--out", required=True, help="Output noisy WAV file.")
    parser.add_argument("--noise-out", help="Optional file to save generated noise to.")
    parser.add_argument("--add-sine", nargs=2, action="append", metavar=("FREQ", "AMP"))
    parser.add_argument("--add-multisine", nargs=2, metavar=("FREQS", "AMPS"))
    parser.add_argument("--add-wind", action="store_true")
    parser.add_argument("--add-ambulance", action="store_true")
    parser.add_argument("--add-gauss", action="store_true")
    parser.add_argument("--snr-db", type=float, default=20.0)
    for noise in NOISES:
        parser.add_argument(f"--add-{noise}", action="store_true")
    args = parser.parse_args()

    signal, sr = librosa.load(args.input, sr=None, mono=True)
    length = len(signal) / sr
    noise_total = np.zeros_like(signal)

    if args.add_sine:
        for freq, amp in args.add_sine:
            n = sine_noise(length, sr, float(freq), float(amp))
            noise_total += n
            print(f"Added sine: {freq} Hz at amplitude {amp}")
    if args.add_multisine:
        freqs = list(map(float, args.add_multisine[0].split(",")))
        amps = list(map(float, args.add_multisine[1].split(",")))
        n = multi_sine_noise(length, sr, freqs, amps)
        noise_total += n
        print(f"Added multi-sine: {freqs} with amps {amps}")
    if args.add_wind:
        wind = wind_noise(length, sr, amp=args.add_wind, cutoff=1500)
        wind = apply_gusts(wind, sr)
        noise_total += wind
        print("Added wind noise")
    if args.add_ambulance:
        n = ambulance_siren(length, sr, amp=args.add_ambulance)
        noise_total += n
        print("Added ambulance siren")
    if args.add_gauss:
        n = gaussian_noise(length, sr, amp=args.add_gauss)
        noise_total += n
        print("Added Gaussian noise")
    for noise in NOISES:
        if args.__dict__[f"add_{noise}"]:
            n = sample_wav(len(signal), os.path.join(DB, f"{noise}.wav"))
            noise_total += n
            print(f"Added {noise} noise")

    output, noise = mix(signal, noise_total, snr_db=args.snr_db)
    if args.noise_out is not None and noise is not None:
        sf.write(args.noise_out, noise, sr)
    sf.write(args.out, output, sr)
    print(f"Saved noisy audio to: {args.out}")


if __name__ == "__main__":
    main()
