# Cohere Transcribe (2B) — Research Notes

> **ROLE: PRIMARY BACKEND.** Used on both Windows and Linux. VAD-chunked inference via HuggingFace transformers.

> **Last updated:** 2026-04-03
> **Model release date:** 2026-03-26
> **Status:** All key facts confirmed from official Hugging Face model card + blog posts. Gaps noted explicitly.

---

## 1. Hugging Face Model ID / Repo

| Field | Value |
|-------|-------|
| **Model ID** | `CohereLabs/cohere-transcribe-03-2026` |
| **HF URL** | https://huggingface.co/CohereLabs/cohere-transcribe-03-2026 |
| **License** | Apache 2.0 |
| **Parameters** | 2 billion |
| **Architecture** | Encoder-decoder X-attention transformer; Fast-Conformer encoder (~90% of params) + lightweight Transformer decoder |
| **ONNX variant** | `onnx-community/cohere-transcribe-03-2026-ONNX` (for transformers.js / WebGPU; not recommended for Python GPU inference) |

**Note:** The Hugging Face org is `CohereLabs`, not `CohereForAI` (which hosts their LLMs).

---

## 2. Python Library / Framework

There are **two** supported paths for local Python inference:

### Path A — Native Transformers (recommended)

Requires `transformers >= 5.4.0`. Uses the built-in `CohereAsrForConditionalGeneration` class.

```python
from transformers import AutoProcessor, CohereAsrForConditionalGeneration
```

### Path B — trust_remote_code (broader compatibility)

Works with `transformers >= 4.56` (but NOT 5.0.x or 5.1.x — weight-loading bug). Uses the generic auto classes with `trust_remote_code=True`.

```python
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, trust_remote_code=True)
```

The `trust_remote_code` path exposes an additional convenience method `model.transcribe()` that handles chunking, batching, and decoding internally.

### Other runtimes

- **vLLM** — for serving over HTTP (OpenAI-compatible `/v1/audio/transcriptions` endpoint)
- **mlx-audio** — Apple Silicon only
- **Cohere API** — free cloud endpoint with rate limits (no local GPU needed)
- **CohereX** (community, `Diffio-AI/CohereX`) — WhisperX-style wrapper adding VAD + word-level alignment

---

## 3. Installation

### Path A — Native Transformers (recommended for new projects)

```bash
pip install "transformers>=5.4.0" torch huggingface_hub soundfile librosa sentencepiece protobuf
```

### Path B — trust_remote_code

```bash
pip install "transformers>=4.56,<5.3,!=5.0.*,!=5.1.*" torch huggingface_hub soundfile librosa sentencepiece protobuf
```

### CUDA requirements

| Item | Details |
|------|---------|
| **Tested with** | `torch==2.10.0` (expected to work with other recent versions) |
| **CUDA toolkit** | Whatever your PyTorch build links against (e.g. CUDA 12.x for recent torch) |
| **VRAM (fp16/bf16)** | ~4-5 GB for weights alone; total ~5-7 GB with KV cache and activations |
| **VRAM (quantized)** | < 8 GB (per Cohere blog; fits on RTX 30-series and 40-series) |
| **VRAM (fp32)** | ~8-10 GB (not recommended; use fp16 or bf16) |
| **16 GB GPU** | Comfortably sufficient. Plenty of headroom. |

> **Windows PyTorch CUDA install:** Make sure to install the CUDA-enabled PyTorch wheel, not the CPU-only default:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu124
> ```

---

## 4. Inference API

### 4a. Minimal example — Native Transformers (Path A)

```python
from transformers import AutoProcessor, CohereAsrForConditionalGeneration
from transformers.audio_utils import load_audio
from huggingface_hub import hf_hub_download

processor = AutoProcessor.from_pretrained("CohereLabs/cohere-transcribe-03-2026")
model = CohereAsrForConditionalGeneration.from_pretrained(
    "CohereLabs/cohere-transcribe-03-2026",
    device_map="auto",
)

audio_file = hf_hub_download(
    repo_id="CohereLabs/cohere-transcribe-03-2026",
    filename="demo/voxpopuli_test_en_demo.wav",
)
audio = load_audio(audio_file, sampling_rate=16000)

inputs = processor(audio, sampling_rate=16000, return_tensors="pt", language="en")
inputs.to(model.device, dtype=model.dtype)

outputs = model.generate(**inputs, max_new_tokens=256)
text = processor.decode(outputs, skip_special_tokens=True)
print(text)
```

### 4b. Minimal example — trust_remote_code (Path B)

```python
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from huggingface_hub import hf_hub_download

model_id = "CohereLabs/cohere-transcribe-03-2026"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, trust_remote_code=True
).to(device)
model.eval()

audio_file = hf_hub_download(
    repo_id=model_id,
    filename="demo/voxpopuli_test_en_demo.wav",
)

texts = model.transcribe(
    processor=processor,
    audio_files=[audio_file],
    language="en",
)
print(texts[0])
```

### 4c. Transcribing a numpy array directly (Path B)

```python
import numpy as np

audio_array = np.random.randn(16000 * 5).astype(np.float32)  # 5 sec example
sample_rate = 16000

texts = model.transcribe(
    processor=processor,
    audio_arrays=[audio_array],
    sample_rates=[sample_rate],
    language="en",
)
print(texts[0])
```

### 4d. Long-form transcription (55+ minutes)

The processor auto-chunks long audio into ~35-second overlapping windows:

```python
inputs = processor(audio=audio_array, sampling_rate=sr, return_tensors="pt", language="en")
audio_chunk_index = inputs.get("audio_chunk_index")
inputs.to(model.device, dtype=model.dtype)

outputs = model.generate(**inputs, max_new_tokens=256)
text = processor.decode(
    outputs, skip_special_tokens=True,
    audio_chunk_index=audio_chunk_index, language="en"
)[0]
```

### 4e. Punctuation control

```python
inputs_pnc   = processor(audio, sampling_rate=16000, return_tensors="pt", language="en", punctuation=True)
inputs_nopnc = processor(audio, sampling_rate=16000, return_tensors="pt", language="en", punctuation=False)
```

### 4f. torch.compile for faster throughput (Path B)

```python
texts = model.transcribe(
    processor=processor,
    audio_arrays=audio_arrays,
    sample_rates=sample_rates,
    language="en",
    compile=True,
    batch_size=16,
)
```

### model.transcribe() API reference (Path B only)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `processor` | `AutoProcessor` | required | Processor instance |
| `language` | `str` | required | ISO 639-1 code (no auto-detection!) |
| `audio_files` | `list[str]` | `None` | File paths (mutually exclusive with `audio_arrays`) |
| `audio_arrays` | `list[np.ndarray]` | `None` | 1-D numpy float arrays (raw waveforms) |
| `sample_rates` | `list[int]` | `None` | Per-array sample rate |
| `punctuation` | `bool` | `True` | Include punctuation |
| `batch_size` | `int` | from config | GPU batch size |
| `compile` | `bool` | `False` | torch.compile encoder |
| `pipeline_detokenization` | `bool` | `False` | Overlap CPU detokenization with GPU (**NOT supported on Windows**) |

**Returns:** `list[str]` — one transcription string per input audio.

---

## 5. Streaming / Chunked Inference

### Does it support true real-time streaming?

**No, not natively.** Cohere Transcribe is designed for **offline / batch inference**. There is no built-in streaming API that accepts a continuous microphone stream and emits partial hypotheses.

### Feeding short chunks

You CAN feed 1-3 second chunks by calling inference repeatedly, but:
- Very short clips (~1-2 sec) may produce lower-quality results.
- There is no mechanism for maintaining decoder state across calls (each call is independent).
- The model's internal chunking targets ~35-second windows.

### Practical approach for live transcription

1. **Accumulate audio** in a rolling buffer (3-5 seconds).
2. **Run inference** on the buffer each time it fills.
3. **Concatenate results** with overlap detection to avoid duplicated words.
4. **Prepend a VAD** (Silero VAD, FireRedVAD) to avoid hallucinations on silence.

### TextStreamer for token-by-token output

You can use transformers' `TextIteratorStreamer` to stream token output during decoding of a single clip:

```python
from transformers import TextIteratorStreamer
from threading import Thread

streamer = TextIteratorStreamer(processor.tokenizer, skip_special_tokens=True)
generation_kwargs = dict(**inputs, max_new_tokens=256, streamer=streamer)
thread = Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()

for partial_text in streamer:
    print(partial_text, end="", flush=True)
thread.join()
```

**Caveat:** This streams token generation for a single already-encoded audio clip — it does NOT accept a continuous audio stream.

### CohereX for VAD-segmented processing

```python
import coherex
model = coherex.load_model("cohere-transcribe-03-2026", device="cuda", vad_method="firered")
audio = coherex.load_audio("audio.mp3")
result = model.transcribe(audio, batch_size=8, chunk_size=30.0)
```

---

## 6. Audio Format Requirements

| Property | Requirement |
|----------|-------------|
| **Sample rate** | 16 kHz (auto-resampled if different) |
| **Channels** | Mono (stereo is auto-averaged to mono) |
| **dtype** | float32 numpy array for `audio_arrays`; any soundfile/librosa format for files |
| **Max single chunk** | ~35 seconds internally (longer audio auto-chunked with overlap) |
| **Supported file formats** | WAV, FLAC, MP3, MPEG, MPGA, OGG |
| **Spectrogram** | Internally converted to log-Mel spectrogram |

---

## 7. Output Format

### Base model

Returns **plain text only**:
- `list[str]` from `model.transcribe()` (Path B)
- A string from `processor.decode()` (Path A)
- With `punctuation=True` (default): capitalized text with punctuation
- With `punctuation=False`: lowercase, no punctuation

**Does NOT provide:** timestamps, speaker diarization, confidence scores, or word-level alignments.

### CohereX (community wrapper) — adds word-level timestamps

```python
import coherex
model = coherex.load_model("cohere-transcribe-03-2026", device="cuda", vad_method="firered")
audio = coherex.load_audio("audio.mp3")
result = model.transcribe(audio, batch_size=8)

model_a, metadata = coherex.load_align_model(
    language_code=result["language"], device="cuda", backend="wav2vec2"
)
result = coherex.align(
    transcript=result["segments"], model=model_a,
    align_model_metadata=metadata, audio=audio, device="cuda"
)
# result["segments"] now has word-level timestamps
```

---

## 8. Known Gotchas

### Windows-specific

1. **`pipeline_detokenization=True` does NOT work on Windows** — relies on `fork()`. Use native transformers path (Path A, `>= 5.4.0`) where it is unnecessary.
2. **PyTorch CUDA on Windows** — must install CUDA-enabled wheel explicitly (`pip install torch --index-url https://download.pytorch.org/whl/cu124`).
3. **`torch.compile`** can be problematic on Windows due to Triton backend issues. Use `compile=False` or WSL2.

### General

4. **No automatic language detection** — you MUST specify the language code.
5. **Hallucination on silence/noise** — prepend a VAD to prevent this.
6. **Transformers 5.0.x and 5.1.x are broken** — weight-loading bug. Use `>= 5.4.0` or `>= 4.56, < 5.0`.
7. **Code-switching performs poorly** — expects monolingual audio.
8. **No timestamps or diarization** in the base model.
9. **`max_new_tokens=256`** may truncate long segments if you feed chunks manually.
10. **`sentencepiece` and `protobuf`** are easy-to-miss required dependencies.
11. **First-run download is ~4 GB** for model weights.

---

## 9. Supported Languages (14)

en, fr, de, it, es, pt, el, nl, pl, zh, ja, ko, vi, ar

---

## 10. Performance

| Metric | Value |
|--------|-------|
| Open ASR Leaderboard rank | #1 (as of 2026-03-26) |
| Average WER (English) | 5.42% |
| RTFx | Up to ~525x real-time (offline, GPU) |

---

## 11. Summary for Desktop App Integration

For a Windows desktop app with a 16 GB NVIDIA GPU:

- **Use Path A** (`transformers >= 5.4.0` + `CohereAsrForConditionalGeneration`)
- **VRAM:** ~5-7 GB at fp16. 16 GB card is more than sufficient.
- **For live transcription:** VAD + rolling 3-5 sec buffer + repeated inference calls
- **For word timestamps:** Add CohereX alignment pipeline
- **Avoid:** `pipeline_detokenization`, `torch.compile` on native Windows

---

## Sources

- [HF Model Card](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)
- [HF Blog: Introducing Cohere-transcribe](https://huggingface.co/blog/CohereLabs/cohere-transcribe-03-2026-release)
- [Cohere Blog: Transcribe](https://cohere.com/blog/transcribe)
- [Cohere API Docs](https://docs.cohere.com/reference/create-audio-transcription)
- [CohereX GitHub](https://github.com/Diffio-AI/CohereX)
- [ONNX variant](https://huggingface.co/onnx-community/cohere-transcribe-03-2026-ONNX)
