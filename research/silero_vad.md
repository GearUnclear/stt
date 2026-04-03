# Silero VAD — Research Notes for Desktop STT Integration

> **ROLE: CORE COMPONENT for Cohere backend.** Gates microphone audio into speech segments before sending to Cohere Transcribe. Not needed for Qwen3-ASR (which handles silence natively via its streaming API).

> Researched: 2026-04-02
> Source: [snakers4/silero-vad](https://github.com/snakers4/silero-vad) (v6.2.1, released Feb 2024)
> Purpose: Real-time voice activity detection to gate microphone audio before ASR

---

## 1. Installation

### Pip package (recommended)

```bash
pip install silero-vad
```

As of v6.2.1, onnxruntime is an **optional** dependency.

### Dependencies

| Package | Version Constraint | Notes |
|---|---|---|
| `torch` | `>=1.12.0` | Required. CPU-only build is sufficient. |
| `torchaudio` | `>=0.12.0` | Used for audio I/O only (read/save files). Not needed if you handle audio yourself. |
| `onnxruntime` | `>=1.16.1` | **Optional** since v6.2.1. Install separately for ONNX mode. |

### Audio backend (for torchaudio file I/O only)

If you use `torchaudio` to read/write audio files, you need one of: FFmpeg (`conda install -c conda-forge 'ffmpeg<7'`), sox (`apt-get install sox`), or soundfile (`pip install soundfile`).

For our use case (live microphone via sounddevice), **we do not need torchaudio at all** — we get raw PCM directly from the mic. The only hard dependency is `torch`.

### System requirements

- Python 3.8+
- 1 GB+ RAM
- Modern x86-64 CPU with AVX/AVX2/AVX-512 (or use ONNX runtime for ARM/other)

### Recommended install for our project

```bash
pip install silero-vad torch torchaudio sounddevice numpy
```

---

## 2. Loading the Model

### Method A: Via the `silero_vad` pip package (preferred)

```python
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps, VADIterator

model = load_silero_vad()           # PyTorch JIT model (default)
model = load_silero_vad(onnx=True)  # ONNX model (requires onnxruntime)
```

`load_silero_vad` signature:

```python
def load_silero_vad(onnx=False, opset_version=16):
    # onnx: bool -- use ONNX runtime instead of PyTorch JIT
    # opset_version: int -- 15 or 16 (only relevant when onnx=True)
    # Returns: model object (JIT ScriptModule or OnnxWrapper)
```

### Method B: Via torch.hub

```python
import torch
torch.set_num_threads(1)

model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=True
)

(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
```

**Method A is preferred** because it bundles the model file inside the pip package (no network download at runtime after install). Method B downloads from GitHub on first use.

### Important: set thread count

```python
import torch
torch.set_num_threads(1)  # Single-thread is optimal for this tiny model
```

---

## 3. Real-Time Streaming API

There are two approaches for streaming:

### Approach A: Raw probability per chunk (low-level)

Call `model(chunk, sampling_rate)` directly on each audio frame:

```python
speech_prob = model(torch.from_numpy(audio_float32), 16000).item()
# Returns: float between 0.0 and 1.0
# > 0.5 = speech, < 0.5 = silence (threshold is tunable)
```

You implement your own state machine to decide when speech starts/ends.

### Approach B: VADIterator (recommended for streaming)

`VADIterator` is a built-in stateful wrapper that tracks speech boundaries:

```python
from silero_vad import load_silero_vad, VADIterator

model = load_silero_vad()

vad_iterator = VADIterator(
    model,
    threshold=0.5,                # Speech probability threshold (default 0.5)
    sampling_rate=16000,           # 8000 or 16000
    min_silence_duration_ms=100,   # Min silence before ending a speech segment
    speech_pad_ms=30               # Padding added to each side of speech chunks
)

# Process audio chunks in a loop:
for chunk in audio_chunks:
    result = vad_iterator(chunk, return_seconds=True)
    # Returns one of:
    #   {'start': 1.234}   -- speech just started at this timestamp
    #   {'end': 5.678}     -- speech just ended at this timestamp
    #   None               -- no state change

# Reset between separate audio streams:
vad_iterator.reset_states()
```

### VADIterator internal state machine

The iterator maintains three state variables:

- `triggered` (bool): currently inside a speech region
- `temp_end` (int): sample position where potential silence started
- `current_sample` (int): running total of processed samples

**Onset detection**: When `speech_prob >= threshold` and not currently triggered, fires `{'start': ...}`.

**Offset detection**: When `speech_prob < (threshold - 0.15)` and currently triggered, starts a silence timer. If silence exceeds `min_silence_duration_ms`, fires `{'end': ...}`. If speech resumes before that, the timer resets (prevents choppy splitting on brief pauses).

---

## 4. Audio Format Requirements

| Parameter | Requirement |
|---|---|
| **Sample rate** | **16000 Hz** (recommended) or 8000 Hz. Higher rates (32k, 48k) are internally downsampled to 16kHz by the JIT model. |
| **Channels** | **Mono** (single channel). Convert stereo to mono before feeding. |
| **Chunk size** | **512 samples** at 16kHz, or **256 samples** at 8kHz. Both equal exactly **32 ms**. This is the ONLY supported chunk size for streaming (fixed since v5.0). |
| **Data type (model input)** | `torch.Tensor` of `float32`, values in range [-1.0, 1.0]. |
| **Data type (from mic)** | Typically `int16` (paInt16). Must be converted to float32. |

### Converting int16 mic audio to float32 tensor

```python
import numpy as np
import torch

def int2float(sound):
    """Convert int16 PCM audio to float32 normalized to [-1, 1]."""
    abs_max = np.abs(sound).max()
    sound = sound.astype('float32')
    if abs_max > 0:
        sound *= 1 / 32768
    sound = sound.squeeze()
    return sound

# Usage:
audio_int16 = np.frombuffer(raw_bytes, np.int16)
audio_float32 = int2float(audio_int16)
tensor = torch.from_numpy(audio_float32)
```

**Critical**: The chunk fed to the model must be exactly 512 samples (at 16kHz). This is a fixed window size as of v5.0+.

---

## 5. Integration Pattern: VAD as a Gate Before ASR

### Recommended architecture

```
Microphone (sounddevice)
    |
    v
[512-sample chunks, 16kHz, mono, float32]
    |
    v
Silero VAD (VADIterator)
    |
    +-- speech_start --> begin accumulating audio into buffer
    +-- speech_end   --> send accumulated buffer to ASR (Whisper, etc.)
    +-- None         --> if triggered, keep accumulating; if not, discard
    |
    v
ASR Model (Whisper / faster-whisper / etc.)
    |
    v
Transcription text
```

### Collecting speech segments

```python
import numpy as np

speech_buffer = []
is_speaking = False

for chunk in mic_stream:
    result = vad_iterator(chunk, return_seconds=False)

    if result and 'start' in result:
        is_speaking = True
        speech_buffer = []

    if is_speaking:
        speech_buffer.append(chunk)

    if result and 'end' in result:
        is_speaking = False
        full_segment = torch.cat(speech_buffer)
        # Send to ASR
        transcription = transcribe(full_segment)
        speech_buffer = []
```

### Tuning for natural speech segmentation

- **`min_silence_duration_ms=100`** (default): Very responsive. Splits on short pauses. Good for command-style input.
- **`min_silence_duration_ms=300-500`**: Better for conversational speech. Waits longer before declaring end-of-speech, producing longer, more complete utterances.
- **`speech_pad_ms=30`** (default): Adds 30ms padding to avoid clipping speech edges. Increase to 50-100ms if ASR clips the beginning of words.
- **`threshold=0.5`** (default): Lower (0.3-0.4) = more sensitive. Higher (0.6-0.7) = stricter.

### WhisperX reference

In WhisperX, Silero VAD segments audio into 3-30 second chunks with 300ms silence padding before sending to Whisper. Reported end-to-end latency: 380-520ms with 95th percentile accuracy, 4x speed improvement over vanilla Whisper.

---

## 6. Performance

### Inference speed

| Metric | Value |
|---|---|
| **Time per chunk (32ms audio)** | **< 1 ms** on a single CPU thread |
| **Real-time factor** | ~30x faster than real-time |
| **Batch benchmark (ONNX)** | 5071s of audio in 6.4s (~800x real-time) |
| **Model size** | ~2 MB (JIT model contains both 8kHz and 16kHz models) |

### CPU vs GPU

**CPU only.** The model is designed for single-thread CPU inference. It is a tiny quantized RNN — GPU would add overhead with no benefit.

```python
torch.set_num_threads(1)  # Always set this. More threads = slower for this model.
```

### PyTorch JIT vs ONNX

- **JIT (default)**: Good enough. < 1ms per chunk.
- **ONNX**: Was historically 4-5x faster. As of v5+, JIT is comparably fast. ONNX may still be ~10% faster.
- **ONNX advantage**: Smaller runtime footprint if you don't need full PyTorch for anything else.

### Latency analysis for real-time use

- Audio chunk duration: 32ms
- VAD inference: < 1ms
- **Total per-chunk latency: ~33ms** (negligible)
- Dominant latency: `min_silence_duration_ms` (how long to wait to confirm speech ended). At 300ms, you add 300ms before sending to ASR.

---

## 7. Windows Compatibility

### Status: Works, with notes

**PyTorch JIT model**: Works on Windows with standard `pip install torch`. No known issues.

**ONNX model**: Some users report `onnxruntime` issues on Windows. If ONNX fails, fall back to JIT (the default). Some users report WSL works when native Windows does not, but this is edge-case.

**PyAudio on Windows**: Can be difficult to install (`pip install pyaudio` may fail). **Use `sounddevice` instead** — it installs cleanly on Windows and has equivalent functionality.

**sounddevice on Windows**: Works reliably. `pip install sounddevice` ships with PortAudio binaries, no compilation needed.

**No known Silero-VAD-specific Windows bugs.** The model itself is platform-agnostic. Issues are confined to audio capture (PyAudio) and occasionally ONNX runtime.

---

## 8. Stateful Model — Important Notes

Silero VAD is an **RNN (LSTM-based) model with internal state**:

- The model "remembers" previous chunks. Each call updates LSTM hidden states.
- **Chunks must be processed sequentially** in arrival order.
- **Call `model.reset_states()` or `vad_iterator.reset_states()`** when starting a new stream, switching speakers, or after long pauses.
- **Do NOT share a single model instance across concurrent audio streams** without resetting. Use separate instances for concurrent streams.

Internal state tensors: `_state` (LSTM hidden, shape `[2, batch, 128]`), `_context` (previous chunk window), `_last_sr`, `_last_batch_size`.

---

## 9. Complete Code Example: Streaming VAD with sounddevice

```python
"""
Minimal real-time Voice Activity Detection using Silero VAD + sounddevice.
Detects speech segments from the microphone and prints start/end events.
Collects speech audio and could forward it to an ASR model.

Requirements:
    pip install silero-vad torch sounddevice numpy
"""

import numpy as np
import torch
import sounddevice as sd
import queue
import sys

# --- Configuration ---
SAMPLE_RATE = 16000
CHUNK_SAMPLES = 512          # Fixed: 512 samples at 16kHz = 32ms
CHANNELS = 1
VAD_THRESHOLD = 0.5
MIN_SILENCE_MS = 300         # Wait 300ms of silence before ending speech
SPEECH_PAD_MS = 30

# --- Load Silero VAD ---
torch.set_num_threads(1)

from silero_vad import load_silero_vad, VADIterator

model = load_silero_vad()
vad_iterator = VADIterator(
    model,
    threshold=VAD_THRESHOLD,
    sampling_rate=SAMPLE_RATE,
    min_silence_duration_ms=MIN_SILENCE_MS,
    speech_pad_ms=SPEECH_PAD_MS,
)


def int2float(sound: np.ndarray) -> np.ndarray:
    """Convert int16 PCM to float32 in [-1, 1] range."""
    sound = sound.astype(np.float32)
    sound /= 32768.0
    return sound


# --- Audio queue for cross-thread communication ---
audio_queue = queue.Queue()


def audio_callback(indata, frames, time_info, status):
    """Called by sounddevice for each audio block."""
    if status:
        print(f"[sounddevice status]: {status}", file=sys.stderr)
    audio_queue.put(indata.copy())


def main():
    print("Loading Silero VAD model... done.")
    print(f"Listening on microphone (16kHz, mono, {CHUNK_SAMPLES}-sample chunks)")
    print("Speak to see VAD events. Ctrl+C to stop.\n")

    speech_buffer = []
    is_speaking = False
    segment_count = 0

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype='int16',
        blocksize=CHUNK_SAMPLES,
        callback=audio_callback,
    )

    try:
        with stream:
            while True:
                audio_int16 = audio_queue.get()
                audio_float32 = int2float(audio_int16.squeeze())
                tensor = torch.from_numpy(audio_float32)

                result = vad_iterator(tensor, return_seconds=True)

                if result is not None:
                    if 'start' in result:
                        print(f"  >> Speech START at {result['start']:.2f}s")
                        is_speaking = True
                        speech_buffer = []

                    if 'end' in result:
                        print(f"  << Speech END   at {result['end']:.2f}s")
                        is_speaking = False
                        segment_count += 1

                        if speech_buffer:
                            full_segment = torch.cat(speech_buffer)
                            duration = len(full_segment) / SAMPLE_RATE
                            print(f"     Segment #{segment_count}: {duration:.2f}s "
                                  f"({len(full_segment)} samples)")
                            # HERE: send full_segment to your ASR model

                        speech_buffer = []

                if is_speaking:
                    speech_buffer.append(tensor.clone())

    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
```

### Full parameter reference for `get_speech_timestamps` (batch mode)

```python
get_speech_timestamps(
    audio: torch.Tensor,
    model,
    threshold: float = 0.5,
    sampling_rate: int = 16000,
    min_speech_duration_ms: int = 250,
    max_speech_duration_s: float = float('inf'),
    min_silence_duration_ms: int = 100,
    speech_pad_ms: int = 30,
    return_seconds: bool = False,
    visualize_probs: bool = False,
    progress_tracking_callback = None,
    neg_threshold: float = None,           # default: threshold - 0.15
    window_size_samples: int = 512,        # 512 (16kHz) or 256 (8kHz)
)
# Returns: [{'start': int/float, 'end': int/float}, ...]
```

---

## 10. Summary / Recommendations

| Decision | Recommendation |
|---|---|
| **Install** | `pip install silero-vad torch sounddevice numpy` |
| **Model loading** | `load_silero_vad()` from pip package (JIT, not ONNX) |
| **Audio capture** | `sounddevice` (not PyAudio — cleaner Windows install) |
| **Sample rate** | 16000 Hz |
| **Chunk size** | 512 samples (32ms) |
| **Streaming API** | `VADIterator` for speech start/end events |
| **min_silence_duration_ms** | Start with 300ms for conversational speech |
| **Threading** | `torch.set_num_threads(1)` always |
| **ASR integration** | Accumulate audio during speech, send segment on speech end |
| **Windows** | No known issues with JIT model + sounddevice |

---

## Sources

- [snakers4/silero-vad (GitHub)](https://github.com/snakers4/silero-vad)
- [silero-vad on PyPI](https://pypi.org/project/silero-vad/)
- [PyTorch Hub: Silero VAD](https://pytorch.org/hub/snakers4_silero-vad_vad/)
- [Wiki: Examples and Dependencies](https://github.com/snakers4/silero-vad/wiki/Examples-and-Dependencies)
- [Wiki: FAQ](https://github.com/snakers4/silero-vad/wiki/FAQ)
