# STT - Speech-to-Text Transcription App

A GUI-based Windows desktop application for speech transcription powered by the best open-source models available. Everything runs locally — no cloud services, no API keys, no subscriptions.

## Features

- **Real-time microphone transcription** — speak and see text appear live
- **File transcription** — drag and drop audio/video files for batch transcription
- **Multiple model support** — switch between top-ranked open-source ASR models
- **GPU acceleration** — CUDA support for fast inference on NVIDIA GPUs
- **Export options** — save transcripts as TXT, SRT subtitles, or copy to clipboard
- **Language detection** — automatic language identification with manual override

## Models

### Cohere Transcribe (2B) — #1 Open-Source ASR (March 2026)

- **Average WER:** 5.42%
- **LibriSpeech clean:** 1.25% WER
- **Architecture:** Conformer encoder + lightweight Transformer decoder
- **Speed:** 525x real-time processing
- **License:** Apache 2.0
- **Notes:** Current benchmark leader for English. Struggles somewhat with Portuguese and German relative to competitors.

### NVIDIA Canary Qwen 2.5B — #2 Open-Source ASR

- **Average WER:** 5.63%
- **LibriSpeech clean:** 1.6% WER
- **Architecture:** Speech-Augmented Language Model (SALM) — FastConformer encoder + Qwen3-1.7B LLM decoder
- **Training data:** 234,000 hours of English speech
- **Noise resilience:** 2.41% WER at 10dB SNR
- **VRAM:** ~8 GB
- **License:** CC-BY-4.0 (commercial use permitted)

### Qwen3-ASR-1.7B (Alibaba, January 2026) — #3

- **Average WER:** 5.76%
- **TED-LIUM:** 4.50% WER (vs Whisper 6.84%, GPT-4o 7.69%)
- **Languages:** 52 languages including 22 Chinese dialects
- **License:** Apache 2.0

## Tech Stack

- **Language:** Python
- **GUI Framework:** PyQt6
- **Audio Processing:** PyAudio, sounddevice
- **Packaging:** PyInstaller (standalone Windows .exe)

## Requirements

- Windows 10/11
- Python 3.10+
- NVIDIA GPU recommended (~8 GB VRAM for largest models, CPU fallback supported)

## Getting Started

```bash
git clone https://github.com/GearUnclear/stt.git
cd stt
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

## License

MIT
