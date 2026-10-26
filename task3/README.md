# Task 3: MIDI Music Generation with an LSTM

This project reads MIDI files, selects the part with the most note events,
quantizes note/rest durations, and trains a PyTorch LSTM to predict the next
note token. The generated sequence is written as a `.mid` file.

## Setup (Windows PowerShell)

Run these commands from the `task3` folder:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If `py` is not recognized, create the environment with `python -m venv .venv`.
Installing PyTorch can take a while; keep an internet connection available
during installation.

## Add MIDI training data

Copy `.mid` or `.midi` files you have permission to use into `data\midi`.
Subfolders are scanned too. Use public-domain, self-created, or appropriately
licensed MIDI files; this project does not download music for you.

To create original sample MIDI files and check the complete pipeline:

```powershell
.\.venv\Scripts\python.exe music_generator.py make-demo-data
```

These short sample melodies verify the software, but a varied, licensed MIDI
collection will produce more interesting results.

## Run without Copilot

After setup, open PowerShell or the VS Code terminal in this folder and run:

```powershell
.\.venv\Scripts\python.exe music_generator.py train --epochs 30
.\.venv\Scripts\python.exe music_generator.py generate
```

The generated file is `output\generated.mid`. Copilot does not need to be
running; the commands run locally on your computer.

The trained model is saved to `output\music_lstm.pt`; the generated MIDI is
saved to `output\generated.mid`. You can open the MIDI file in a MIDI player or
music editor. To render it to audio, use a MIDI application with a soundfont
or install a MIDI synthesizer such as FluidSynth separately.

Optional generation settings include `--notes 128`, `--temperature 0.8`, and
`--tempo 110`. A lower temperature makes predictions less random. The model is
trained from scratch and does not use a pretrained or external AI service.
