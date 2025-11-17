# Denoise

Scripts for testing denoising signal processing techniques on environmental
audio. Supporting the CIROH research project at UVM.

## Setup

1. Install UV package manager.
2. `uv run` the script you want to run!

## `spectrogram.py`

Generate spectrogram for a given .wav file.

## `single_channel.py`

Single channel noise removal techniques using amplitude threshold and filtering techniques.

## `dual_channel.py`

Dual channel noise removal techniques using spectral subtraction.

## `add_noise.py`

Script to programatically add predefined noise profiles to WAV files to make testing
noise removal easier. Can have it automatically generate new options by dropping
additional .wav files into the `noisedb` directory.
