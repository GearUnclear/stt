# STT - Speech-to-Text Transcription App

A GUI-based Windows desktop application for speech transcription powered by the best open-source models available. Everything runs locally — no cloud services, no API keys, no subscriptions.

## Features

- **Live transcription** — chunked real-time transcription from your microphone with sub-200ms perceived latency
- **File transcription** — drag and drop audio/video files for batch transcription
- **Multiple model support** — switch between top-ranked open-source ASR models
- **GPU acceleration** — CUDA support for fast inference on NVIDIA GPUs (16 GB VRAM target)
- **Export options** — save transcripts as TXT, SRT subtitles, or copy to clipboard
- **Language detection** — automatic language identification with manual override
- **VAD-powered chunking** — Silero VAD splits audio on silence/pauses for clean chunk boundaries, avoiding cut-off words and repeated phrases

## How Live Transcription Works

These models are encoder-decoder architectures — they need a chunk of audio to produce output, not a continuous stream. The app uses **chunked real-time transcription**, which is the standard approach and what users perceive as "live":

1. **Capture** audio in rolling 1-3 second windows via sounddevice
2. **VAD gating** — Silero VAD detects speech boundaries so chunks split on natural pauses, not arbitrary time cuts
3. **Transcribe** each chunk on the GPU — Cohere Transcribe processes a 2s chunk in ~4ms (525x real-time)
4. **Display** results as they arrive, stitching transcript together with overlap deduplication
5. **Result:** sub-200ms perceived latency that feels instant

### VRAM Budget (16 GB GPU)

| Model | ~VRAM (FP16) | Headroom |
|-------|-------------|----------|
| Cohere Transcribe 2B | ~4-5 GB | ~11 GB free |
| NVIDIA Canary Qwen 2.5B | ~8 GB | ~8 GB free |
| Qwen3-ASR 1.7B | ~3-4 GB | ~12 GB free |

All three models fit comfortably. Enough headroom to keep two models loaded simultaneously for fast switching.

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
- **Audio Processing:** sounddevice
- **VAD:** Silero VAD (silence/pause detection for clean chunk boundaries)
- **Packaging:** PyInstaller (standalone Windows .exe)

## Requirements

- Windows 10/11
- Python 3.10+
- NVIDIA GPU with 16 GB VRAM (e.g. RTX 4060 Ti 16GB, RTX 4080/4090, RTX 3090)

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
