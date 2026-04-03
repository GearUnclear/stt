# STT - Speech-to-Text Transcription App

A local-first desktop application for live speech transcription. No cloud services, no API keys, no subscriptions — everything runs on your GPU.

**Primary backend:** Cohere Transcribe 2B (#1 open-source ASR, March 2026)
**Secondary backend:** Qwen3-ASR 1.7B (52-language support, native streaming)

## Platform Strategy

| Platform | Backend | Streaming Method | Status |
|----------|---------|-----------------|--------|
| **Windows** | Cohere Transcribe 2B | VAD-chunked inference via transformers | Primary target |
| **Linux** | Qwen3-ASR 1.7B | Native streaming via vLLM | Secondary target |
| **Linux** | Cohere Transcribe 2B | VAD-chunked inference via transformers | Also supported |

Windows gets Cohere (best English accuracy, native Windows support). Linux gets both Cohere and Qwen3-ASR (Qwen3's native streaming API requires vLLM, which is Linux-only).

## Features

- **Live transcription** — speak and see text appear in real-time
- **File transcription** — drag and drop audio/video files for batch transcription
- **GPU acceleration** — CUDA inference on NVIDIA GPUs (16 GB VRAM target)
- **Export options** — save transcripts as TXT, SRT subtitles, or copy to clipboard
- **VAD-powered chunking** — Silero VAD splits audio on natural pauses for clean transcription boundaries

## How Live Transcription Works

### Cohere (Windows + Linux) — VAD-chunked inference

1. **Capture** audio from microphone via sounddevice (16 kHz, mono)
2. **VAD gating** — Silero VAD detects speech start/end so chunks split on natural pauses
3. **Transcribe** each speech segment on the GPU — Cohere processes a 2s chunk in ~4ms (525x real-time)
4. **Display** results as they arrive, stitching transcript together
5. **Result:** sub-200ms perceived latency

### Qwen3-ASR (Linux only) — native streaming

1. **Capture** audio from microphone via sounddevice (16 kHz, mono)
2. **Stream** 500ms audio steps directly to Qwen3's streaming API (no external VAD needed)
3. **Display** incrementally with prefix rollback (last ~5 tokens may revise as context grows)
4. **Result:** true token-by-token streaming with self-correction

## Models

### Cohere Transcribe (2B) — Primary

- **Model ID:** `CohereLabs/cohere-transcribe-03-2026`
- **Average WER:** 5.42% (#1 on Open ASR Leaderboard)
- **LibriSpeech clean:** 1.25% WER
- **Architecture:** Conformer encoder + lightweight Transformer decoder
- **Speed:** 525x real-time
- **VRAM:** ~5-7 GB (FP16)
- **Languages:** 14 (must specify — no auto-detection)
- **License:** Apache 2.0
- **Install:** `pip install "transformers>=5.4.0" torch`

### Qwen3-ASR 1.7B — Secondary (Linux)

- **Model ID:** `Qwen/Qwen3-ASR-1.7B`
- **Average WER:** 5.76%
- **TED-LIUM:** 4.50% WER (vs Whisper 6.84%)
- **Architecture:** Conv2D stem + Transformer encoder + Qwen3 LLM decoder
- **VRAM:** ~4-6 GB (BF16)
- **Languages:** 52 with auto-detection (97.9% accuracy)
- **Timestamps:** Word-level via companion `Qwen3-ForcedAligner-0.6B`
- **License:** Apache 2.0
- **Install:** `pip install "qwen-asr[vllm]"`

### VRAM Budget (16 GB GPU)

| Model | ~VRAM | Headroom |
|-------|-------|----------|
| Cohere Transcribe 2B | ~5-7 GB | ~9-11 GB free |
| Qwen3-ASR 1.7B | ~4-6 GB | ~10-12 GB free |

Both fit comfortably with room to spare.

## Tech Stack

- **Language:** Python
- **GUI Framework:** PyQt6
- **Audio Capture:** sounddevice
- **VAD:** Silero VAD (Cohere path — silence/pause detection for clean chunk boundaries)
- **ASR — Cohere:** HuggingFace transformers (`CohereAsrForConditionalGeneration`)
- **ASR — Qwen3:** qwen-asr + vLLM (native streaming)
- **Packaging:** PyInstaller (Windows .exe), AppImage or pip (Linux)

## Requirements

- Python 3.12+
- NVIDIA GPU with 16 GB VRAM (e.g. RTX 4060 Ti 16GB, RTX 4080/4090, RTX 3090)
- Windows 10/11 or Linux
- CUDA 12.x+

## Getting Started

```bash
git clone https://github.com/GearUnclear/stt.git
cd stt
python -m venv venv

# Windows
venv\Scripts\activate

# Linux
source venv/bin/activate

pip install -r requirements.txt
python main.py
```

## License

MIT
