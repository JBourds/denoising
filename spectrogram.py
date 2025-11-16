import sys
import wave

import matplotlib.pyplot as plt
import numpy as np

EIGHT_BIT_BIAS = 128


def plot_wav_analysis(
    filename: str,
):
    """
    Plots the time and frequency domain of a given WAV file.

    Args:
        filename (str): The path to the WAV file.
    """
    try:
        with wave.open(filename, "rb") as wf:
            samp_width = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            frames = wf.readframes(n_frames)

        # 16-bit
        if samp_width == 2:
            audio_data = np.frombuffer(frames, dtype=np.int16)
        # 8-bit
        elif samp_width == 1:
            audio_data = np.frombuffer(frames, dtype=np.uint8) - EIGHT_BIT_BIAS
        else:
            print(
                f"Unsupported sample width: {samp_width} bytes. Only 8-bit and 16-bit supported."
            )
            return

        # Time domain
        time = np.linspace(0, n_frames / framerate, num=len(audio_data))
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(time, audio_data)
        plt.title(f"Time Domain of {filename}")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)

        # Frequency domain
        yf = np.fft.fft(audio_data)
        xf = np.fft.fftfreq(len(audio_data), 1 / framerate)

        # Take the real frequency side and magnitude
        xf_positive = xf[: len(audio_data) // 2]
        yf_positive = 2.0 / n_frames * np.abs(yf[0 : len(audio_data) // 2])

        plt.subplot(2, 1, 2)
        plt.plot(xf_positive, yf_positive)
        plt.title(f"Frequency Domain of {filename}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Too few arguments.\n\tUsage: python3 plot.py <filename.wav>")
    plot_wav_analysis(sys.argv[1])
