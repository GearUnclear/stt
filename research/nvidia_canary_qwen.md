# NVIDIA Canary-Qwen-2.5B — Research Notes

> **DECISION: DROPPED.** Not being used in this project. Requires WSL2/NeMo, heaviest dependency tree, English-only, and less accurate than Cohere. Kept for reference only.

> **Last updated:** 2026-04-02
> **Model release date:** July 2025
> **Status:** Confirmed from HuggingFace model card, NeMo docs, and HF Spaces demo code.

---

## 1. Hugging Face Model ID / Repo

| Field | Value |
|-------|-------|
| **Model ID** | `nvidia/canary-qwen-2.5b` |
| **HF URL** | https://huggingface.co/nvidia/canary-qwen-2.5b |
| **License** | CC-BY-4.0 (commercial use permitted) |
| **Parameters** | 2.5 billion |
| **Architecture** | SALM (Speech-Augmented Language Model) — FastConformer encoder + Qwen3-1.7B LLM decoder |
| **File size** | 5.12 GB safetensors |
| **Language** | English only |

---

## 2. Python Library / Framework

Requires **NVIDIA NeMo**, specifically the new `speechlm2` collection:

```python
from nemo.collections.speechlm2.models import SALM
```

**Important:** This is the NEW `speechlm2` collection, NOT the older `nemo.collections.asr`. Old Canary-1B scripts and APIs are incompatible.

---

## 3. Installation

### From git (required — not on PyPI stable)

```bash
pip install "nemo_toolkit[asr,tts] @ git+https://github.com/NVIDIA/NeMo.git"
```

### Requirements

| Item | Details |
|------|---------|
| **Python** | 3.12+ |
| **PyTorch** | 2.6+ |
| **CUDA** | 12.x+ |
| **VRAM** | ~8 GB at inference (bf16) |
| **Note** | Heavy dependency tree including megatron-core |

### CUDA PyTorch on Windows

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

---

## 4. Inference API

### Loading the model

```python
from nemo.collections.speechlm2.models import SALM

model = SALM.from_pretrained("nvidia/canary-qwen-2.5b")
model = model.bfloat16().eval().to("cuda")
```

### Transcribing a file

```python
result = model.generate(
    prompts=[
        f"Transcribe the following audio:\n{model.audio_locator_tag}"
    ],
    audios=["path/to/audio.wav"],
    audio_lens=None,  # auto-detected from file
    max_new_tokens=256,
)
print(result[0])
```

### Transcribing a tensor

```python
import torch
import torchaudio

waveform, sr = torchaudio.load("audio.wav")
# Resample to 16kHz if needed
if sr != 16000:
    waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

result = model.generate(
    prompts=[f"Transcribe the following audio:\n{model.audio_locator_tag}"],
    audios=[waveform],
    audio_lens=[torch.tensor([waveform.shape[1]])],
    max_new_tokens=256,
)
print(result[0])
```

### Chat-style prompt format

The model uses chat-style prompts with `model.audio_locator_tag` as a placeholder for where audio embedding is inserted.

---

## 5. Streaming / Chunked Inference

### Does it support true streaming?

**No native streaming.** No hidden-state carryover between calls.

### For long audio

Use `lhotse` to split into 40-second non-overlapping windows and batch-process. Reference implementation exists in the HuggingFace Spaces demo `app.py`.

### Practical approach for live transcription

1. VAD-gated segmentation (Silero VAD)
2. Feed segments up to 40 seconds to the model
3. Stitch results together

The old NeMo chunked inference scripts (from Canary-1B era) are incompatible with this model.

---

## 6. Audio Format Requirements

| Property | Requirement |
|----------|-------------|
| **Sample rate** | 16 kHz |
| **Channels** | Mono |
| **File formats** | WAV, FLAC |
| **Max chunk length** | 40 seconds |
| **Tensor shape** | `(batch, samples)` |

---

## 7. Output Format

- **Plain text** with punctuation and capitalization
- **No word-level timestamps** (confirmed "No" by NVIDIA)
- **No confidence scores**
- Token IDs decoded via `model.tokenizer.ids_to_text()`

---

## 8. Known Gotchas

### Windows-specific

1. **NeMo does NOT officially support Windows pip install.** Use WSL2.
2. **Massive dependency tree** with megatron-core — import errors are common on non-Linux systems.

### General

3. **bfloat16 requires Ampere+ GPUs** (RTX 30xx, 40xx, A100, etc.). Older GPUs need fp16.
4. **Old Canary-1B scripts/APIs are incompatible** — the `speechlm2` API is completely different from the old `asr` collection.
5. **numpy < 2.0 conflict** was fixed Dec 2025 but may bite older NeMo installs.
6. **40-second max per chunk** — must split longer audio externally.

---

## 9. Performance

| Metric | Value |
|--------|-------|
| Open ASR Leaderboard rank | #2 |
| Average WER | 5.63% |
| LibriSpeech clean | 1.6% WER |
| Noise resilience (10dB SNR) | 2.41% WER |
| Training data | 234,000 hours English speech |

---

## 10. Summary for Desktop App Integration

For a Windows desktop app with a 16 GB NVIDIA GPU:

- **Requires WSL2** — NeMo does not support native Windows
- **VRAM:** ~8 GB at bf16. Fits on 16 GB with headroom.
- **Best noise resilience** of the three models (2.41% WER at 10dB SNR)
- **English only** — no multilingual support
- **No streaming** — VAD + chunked batch inference
- **Heaviest dependency tree** — NeMo + megatron-core

---

## Sources

- [HF Model Card](https://huggingface.co/nvidia/canary-qwen-2.5b)
- [NVIDIA NeMo GitHub](https://github.com/NVIDIA/NeMo)
- [HF Spaces demo app.py](https://huggingface.co/spaces/nvidia/canary-qwen-2.5b)
