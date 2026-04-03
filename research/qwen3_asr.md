# Qwen3-ASR-1.7B — Research Notes for Desktop Live Transcription

> **ROLE: SECONDARY BACKEND (Linux only).** Native streaming via vLLM. Provides 52-language support and auto language detection. Not used on Windows (vLLM is Linux-only).

> **Last updated:** 2026-04-03
> **Target platform:** Linux desktop, 16 GB NVIDIA GPU, Python

---

## 1. Hugging Face Model ID / Repo

| Model | HF Repo | Parameters | Purpose |
|-------|---------|------------|---------|
| **Qwen3-ASR-1.7B** | `Qwen/Qwen3-ASR-1.7B` | ~2B (listed as 1.7B, 2B in tensor metadata) | Primary ASR model |
| Qwen3-ASR-0.6B | `Qwen/Qwen3-ASR-0.6B` | ~0.6B | Lighter/faster variant |
| Qwen3-ForcedAligner-0.6B | `Qwen/Qwen3-ForcedAligner-0.6B` | ~0.6B | Word-level timestamp alignment |

- **License**: Apache 2.0
- **Released**: January 29, 2026
- **Paper**: arXiv:2601.21337
- **GitHub**: https://github.com/QwenLM/Qwen3-ASR
- **HF Collection**: https://huggingface.co/collections/Qwen/qwen3-asr

The weights are stored in BF16. The 1.7B is ~3.5-4 GB on disk.

---

## 2. Python Library / Framework

The primary interface is the **`qwen-asr`** PyPI package, wrapping two backends:

- **Transformers backend** — HuggingFace transformers under the hood. Simpler setup, offline only (no streaming).
- **vLLM backend** — uses vllm for optimized inference. Required for streaming. Faster batch throughput.

```python
from qwen_asr import Qwen3ASRModel
from qwen_asr import Qwen3ForcedAligner      # for timestamps
from qwen_asr import parse_asr_output         # for parsing raw model output
```

---

## 3. Installation

### Minimal install (transformers backend)

```bash
pip install -U qwen-asr
```

### With vLLM backend (needed for streaming)

```bash
pip install -U "qwen-asr[vllm]"
```

### Optional: FlashAttention 2

```bash
pip install -U flash-attn --no-build-isolation
```

### CUDA requirements

- CUDA-compatible NVIDIA GPU required
- BF16 or FP16 dtype
- FlashAttention 2 is optional — without it, PyTorch SDPA is used (slower but functional)
- Python 3.12 recommended

---

## 4. Inference API

### Minimal example (transformers backend)

```python
import torch
from qwen_asr import Qwen3ASRModel

model = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-1.7B",
    dtype=torch.bfloat16,
    device_map="cuda:0",
    max_inference_batch_size=32,
    max_new_tokens=256,
)

results = model.transcribe(
    audio="path/to/audio.wav",
    language=None,  # None = auto-detect; or "English", "Chinese", etc.
)

print(results[0].language)  # e.g. "English"
print(results[0].text)      # transcribed text
```

### With timestamps (forced aligner)

```python
model = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-1.7B",
    dtype=torch.bfloat16,
    device_map="cuda:0",
    max_inference_batch_size=32,
    max_new_tokens=256,
    forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
    forced_aligner_kwargs=dict(dtype=torch.bfloat16, device_map="cuda:0"),
)

results = model.transcribe(
    audio=["audio1.wav", "audio2.wav"],
    language=["Chinese", "English"],
    return_time_stamps=True,
)

for r in results:
    print(r.language, r.text)
    for ts in r.time_stamps:
        print(f"  {ts.text}  {ts.start_time} -> {ts.end_time}")
```

### Passing a numpy array directly

```python
import numpy as np
import soundfile as sf

wav, sr = sf.read("audio.wav", dtype="float32")
results = model.transcribe(audio=(wav, sr), language=None)
```

### vLLM backend (faster, required for streaming)

```python
if __name__ == '__main__':    # REQUIRED guard for vLLM
    model = Qwen3ASRModel.LLM(
        model="Qwen/Qwen3-ASR-1.7B",
        gpu_memory_utilization=0.7,
        max_inference_batch_size=128,
        max_new_tokens=4096,
    )
    results = model.transcribe(audio="audio.wav", language=None)
    print(results[0].text)
```

---

## 5. Streaming / Chunked Inference

**Status: Supported, but vLLM backend only.** Transformers backend is offline-only.

### Streaming API (from official repo)

```python
import io
import numpy as np
import soundfile as sf
from qwen_asr import Qwen3ASRModel

def main():
    asr = Qwen3ASRModel.LLM(
        model="Qwen/Qwen3-ASR-1.7B",
        gpu_memory_utilization=0.8,
        max_new_tokens=32,       # small value for streaming latency
    )

    state = asr.init_streaming_state(
        unfixed_chunk_num=2,     # trailing chunks that may be revised
        unfixed_token_num=5,     # tokens that may be revised
        chunk_size_sec=2.0,      # encoder chunk size in seconds
    )

    sr = 16000
    step_ms = 500
    step_samples = int(step_ms / 1000.0 * sr)

    wav, orig_sr = sf.read("audio.wav", dtype="float32")

    pos = 0
    while pos < wav.shape[0]:
        chunk = wav[pos : pos + step_samples]
        pos += chunk.shape[0]
        asr.streaming_transcribe(chunk, state)
        print(f"language={state.language!r}  text={state.text!r}")

    asr.finish_streaming_transcribe(state)
    print(f"FINAL: language={state.language!r}  text={state.text!r}")

if __name__ == "__main__":
    main()
```

### Streaming behavior

- 2-second encoder chunks internally
- Last `unfixed_token_num` tokens may be revised as more audio arrives (prefix rollback)
- **No batch inference** during streaming (single-stream only)
- **No timestamps** during streaming
- No external VAD needed — model handles silence natively

---

## 6. Audio Format Requirements

| Property | Requirement |
|----------|-------------|
| **Sample rate** | 16 kHz (auto-resamples other rates) |
| **Channels** | Mono (stereo downmixed automatically) |
| **dtype** | float32 numpy array when passing `(ndarray, sr)` tuples |
| **File formats** | WAV, MP3, MP4, MOV, MKV, M4A via FFmpeg |
| **Max duration** | ~20 minutes per pass (non-streaming); unlimited via streaming |

---

## 7. Output Format

### Basic output

```python
result.language   # str: "English", "Chinese", "Japanese", etc.
result.text       # str: full transcription
```

### With timestamps (Qwen3-ForcedAligner-0.6B)

```python
result.time_stamps     # list of timestamp objects
ts.text                # str: word or character
ts.start_time          # float: seconds
ts.end_time            # float: seconds
```

Forced aligner supports word/character-level timestamps for 11 languages. Audio limit: 5 minutes.

### Language handling

- `language=None` — auto-detect (97.9% accuracy)
- `language="English"` — force a specific language (full English name)
- 52 supported languages/dialects: 30 languages + 22 Chinese dialects

---

## 8. VRAM Estimates for 16 GB GPU

| Configuration | Estimated VRAM | Notes |
|---------------|---------------|-------|
| 1.7B, BF16, transformers | ~4-6 GB | Fits comfortably |
| 1.7B + Aligner 0.6B, BF16 | ~6-9 GB | Still fits |
| 1.7B, vLLM, utilization=0.7 | ~11.2 GB | Fits; adjust down if needed |

---

## 9. Known Gotchas

### Windows-specific

1. **vLLM does not support Windows natively.** This is the biggest obstacle for streaming. Workarounds:
   - **WSL2**: Most practical. Ubuntu on WSL2 with GPU passthrough.
   - **Community fork**: https://github.com/SystemPanic/vllm-windows — requires PyTorch nightly.
   - **Transformers backend works natively on Windows** but has no streaming.

2. **FlashAttention does not build easily on Windows.** Precompiled wheels at https://github.com/bdashore3/flash-attention/releases. FlashAttention is optional.

### General

3. **`if __name__ == '__main__':` guard mandatory with vLLM** (multiprocessing spawn).
4. **`max_new_tokens` must match audio length.** Default 256 for short clips; 2048-4096 for long audio.
5. **Streaming has no timestamps and no batch mode.**
6. **No VAD needed** — model handles silence natively.

---

## 10. Performance

| Benchmark | Qwen3-ASR-1.7B | Whisper-large-v3 |
|-----------|----------------|-------------------|
| Average WER | 5.76% | — |
| Librispeech Clean | 1.63% | 2.74% |
| Librispeech Other | 3.38% | 5.22% |
| TED-LIUM | 4.50% | 6.84% |
| Language ID accuracy | 97.9% | 94.1% |

---

## 11. Summary for Desktop App Integration

For a Windows desktop app with a 16 GB NVIDIA GPU:

- **Transformers backend works natively on Windows** — `pip install qwen-asr`
- **For streaming: needs WSL2** (vLLM dependency)
- **VRAM:** ~4-6 GB at bf16. Plenty of headroom.
- **Best multilingual support** — 52 languages with auto-detection
- **Has native streaming API** (only model of the three that does)
- **Word timestamps available** via companion ForcedAligner model
- **Lightest dependency footprint** of the three

---

## Sources

- [Qwen/Qwen3-ASR-1.7B Model Card](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
- [QwenLM/Qwen3-ASR GitHub](https://github.com/QwenLM/Qwen3-ASR)
- [arXiv:2601.21337](https://arxiv.org/abs/2601.21337)
- [qwen-asr on PyPI](https://pypi.org/project/qwen-asr/)
- [vLLM Qwen3-ASR Guide](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-ASR.html)
- [Streaming example](https://github.com/QwenLM/Qwen3-ASR/blob/main/examples/example_qwen3_asr_vllm_streaming.py)
