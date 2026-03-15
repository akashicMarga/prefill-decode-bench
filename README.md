# prefill-decode-bench

A benchmarking framework for studying transformer inference behaviour across
hardware, focusing on **prefill vs decode phases** for edge AI systems.

Measures **prefill and decode speed** for LLMs and surfaces the
memory-bandwidth-bound nature of autoregressive inference.

Works on **Apple Silicon via MLX**, **Nvidia GPUs via CUDA**, and
**any platform via llama.cpp** (GGUF models — CPU, Metal, CUDA, Vulkan).
Results are saved as JSON and can be plotted and compared across chips.

Companion to: *Running Conversational AI Locally: A Systems View on Memory, Bandwidth, and Hardware Choices*

---

## Structure

```
prefill-decode-bench/
├── profiler.py              # entry point — auto-detects backend
├── plot_results.py          # visualize and compare results
├── setup_llamacpp.sh        # one-time build script for llama.cpp
├── requirements.txt
├── backends/
│   ├── types.py             # shared dataclasses (ProfileRun, etc.)
│   ├── utils.py             # shared prompt builder + summary printer
│   ├── mlx/
│   │   ├── profiler.py      # MLX backend (Apple Silicon)
│   │   └── benchmark_concurrent.py  # LLM + Whisper contention test
│   ├── cuda/
│   │   └── profiler.py      # CUDA backend (Nvidia)
│   └── llamacpp/
│       └── profiler.py      # llama.cpp backend (native CLI)
├── docs/
│   └── research_log/        # weekly experiment notes and findings
│       └── week1.md
├── vendor/
│   └── llama.cpp/           # built from source (git-ignored)
└── results/                 # JSON output + charts (git-ignored)
```

---

## Install

**Apple Silicon (MLX — native):**
```bash
pip install mlx-lm matplotlib
```

**Nvidia GPU (CUDA):**
```bash
# Install PyTorch for your CUDA version first: https://pytorch.org
pip install transformers accelerate matplotlib
```

**llama.cpp (any platform — GGUF models):**
```bash
# One-time: clone and build llama.cpp from source (auto-detects Metal/CUDA)
./setup_llamacpp.sh

# Optional: pull GGUF models from HuggingFace
pip install huggingface-hub matplotlib

# Update for new model support (e.g. after a new architecture release)
./setup_llamacpp.sh --update
```

---

## Usage

```bash
# Auto-detects backend (MLX > CUDA > llama.cpp)
python profiler.py --model mlx-community/Llama-3.2-3B-Instruct-4bit

# Force a specific backend
python profiler.py --backend mlx      --model mlx-community/Llama-3.2-3B-Instruct-4bit
python profiler.py --backend cuda     --model meta-llama/Llama-3.2-3B-Instruct
python profiler.py --backend llamacpp --model ./models/llama-3.2-3b-q4_k_m.gguf

# llama.cpp with a HuggingFace GGUF repo
python profiler.py --backend llamacpp \
    --model unsloth/Qwen3.5-0.8B-GGUF \
    --gguf-file "*Q4_K_M.gguf"

# llama.cpp CPU-only (no GPU offload)
python profiler.py --backend llamacpp --model ./models/model.gguf --gpu-layers 0

# llama.cpp with a custom binary path
python profiler.py --backend llamacpp --model ./models/model.gguf --llamacpp-bin /usr/local/bin/llama-bench

# Save chart to results/
python profiler.py --model mlx-community/Mistral-7B-Instruct-v0.3-4bit --plot

# Custom sweep
python profiler.py --model mlx-community/Llama-3.2-3B-Instruct-4bit \
    --prefill-lengths 128 512 1024 2048 4096 \
    --decode-kv-sizes 64 256 512 1024 2048 \
    --decode-tokens 150 --runs 5 --plot
```

---

## What it measures

**Prefill** — time to process the full input prompt (time to first token).
All tokens are processed in parallel. This phase is **compute-bound** — GPU
utilization is high, throughput per token stays roughly flat as prompt length grows.

**Decode** — time to generate each output token sequentially.
Each step reads all model weights plus the full accumulated KV cache.
This phase is **memory-bandwidth bound** — throughput degrades as the KV
cache grows, because more bytes must be read per token as conversations lengthen.

This distinction is why local inference for long conversational sessions is limited
by memory bandwidth, not raw compute — and why a pipeline that feels fast at turn 5
can feel noticeably slower by turn 40.

---

## Reading the output

```
PREFILL  (compute-bound — all input tokens processed in parallel)
    Tokens       tok/s        ms    ms/tok
  ------------------------------------------
       128      4823.2       27      0.21
       512      5102.4      100      0.20
      1024      4987.1      205      0.20
      2048      4831.6      424      0.21
      4096      4654.8      880      0.21

DECODE   (memory-bandwidth-bound — sequential, reads all weights + KV cache per token)
    KV cache       tok/s    ms/tok
  ----------------------------------
          64        62.3      16.1
         256        58.7      17.0
         512        52.1      19.2
        1024        44.8      22.3
        2048        38.2      26.2

  Decode degradation  KV=64→2048: 38.7%
  Significant — bandwidth is the bottleneck for long conversations.
```

**Prefill tok/s** — should stay relatively flat. Large drops at longer prompts
indicate you're saturating compute during the parallel processing phase.

**Decode tok/s** — will fall as KV cache grows. This is expected and unavoidable;
the question is how fast it falls.

**Decode degradation %** — how much slower decode gets from smallest to largest
KV cache tested. Under 10%: minimal pressure. 10–25%: moderate. Over 25%: bandwidth
is the binding constraint for long sessions on this hardware.

---

## Comparing results

```bash
# Single run
python plot_results.py results/profile_mlx_Apple-M3-Max_....json

# Compare two runs (overlaid on same axes)
python plot_results.py results/profile_mlx_....json results/profile_cuda_....json

# All runs in results/
python plot_results.py results/
```

Comparison mode overlays prefill and decode curves from multiple runs on shared axes —
useful for comparing models on the same chip, or the same model across chips.

---

## Concurrency benchmark (MLX)

Measures the bandwidth contention penalty when LLM decode and Whisper ASR
run simultaneously. This surfaces the jitter problem in speech pipelines:
each model hits its rated throughput in isolation, but both slow under
concurrent load because they share the same memory bus.

```bash
pip install mlx-whisper

python -m backends.mlx.benchmark_concurrent \
    --model mlx-community/Llama-3.2-3B-Instruct-4bit \
    --whisper-model mlx-community/whisper-small

# With a real audio file
python -m backends.mlx.benchmark_concurrent \
    --model mlx-community/Mistral-7B-Instruct-v0.3-4bit \
    --whisper-model mlx-community/whisper-large-v3 \
    --audio recording.wav
```

```
  Bandwidth Contention — MLX backend
                                     Isolated   Concurrent       Δ
  ----------------------------------------------------------------
  LLM decode (tok/s)                     62.3        51.8   -16.8%
  Whisper (real-time factor)             8.20x       6.94x  -15.4%

  → Significant contention (16.8%). Both models are competing for
    the same memory bus. This will produce audible jitter in a real
    speech pipeline.
```

Degradation under ~5%: comfortable for concurrent use.
Degradation over ~15%: will produce jitter in a speech pipeline.

---

## Recommended models

**MLX (mlx-community on HuggingFace):**

| Model | Size | Use |
|-------|------|-----|
| `Llama-3.2-3B-Instruct-4bit` | ~2 GB | Fast testing |
| `Mistral-7B-Instruct-v0.3-4bit` | ~4 GB | Standard 7B baseline |
| `Llama-3.1-8B-Instruct-4bit` | ~5 GB | Current 8B baseline |
| `Llama-3.1-13B-Instruct-4bit` | ~8 GB | 13B on 36GB+ |

**CUDA (any HuggingFace causal LM):**

| Model | Notes |
|-------|-------|
| `meta-llama/Llama-3.2-3B-Instruct` | Fast, needs HF access |
| `mistralai/Mistral-7B-Instruct-v0.3` | Standard baseline |
| `TheBloke/*-GPTQ` | Quantized, lower VRAM |

**llama.cpp (GGUF models — any HuggingFace GGUF repo or local file):**

| Model | Size (Q4_K_M) | Notes |
|-------|---------------|-------|
| `unsloth/Qwen3.5-0.8B-GGUF` | ~533 MB | Tiny, fast testing |
| `bartowski/Llama-3.2-3B-Instruct-GGUF` | ~2 GB | Fast testing, any platform |
| `bartowski/Mistral-7B-Instruct-v0.3-GGUF` | ~4 GB | Standard baseline |
| `bartowski/Llama-3.1-8B-Instruct-GGUF` | ~5 GB | Current 8B baseline |

Use `--gguf-file` to select quantization: `*Q2_K.gguf`, `*Q4_K_M.gguf`, `*Q5_K_M.gguf`, `*Q8_0.gguf`

Because llama.cpp is built from source, new model architectures work immediately
after `./setup_llamacpp.sh --update` — no waiting for pip releases.

---

## Notes on timing methodology

- **MLX**: `mx.eval()` is the sync point — MLX is lazy-evaluated, so timing wraps eval() not the function call
- **CUDA**: `torch.cuda.synchronize()` is the sync point — CUDA ops are async by default
- **llama.cpp**: native `llama-bench` binary runs as a subprocess — built from source, always latest model support
- Median across `--runs` runs is used (not mean) to reduce noise from JIT compilation and GC
- A warmup pass runs before all measurements to trigger kernel compilation
- Decode uses greedy sampling (argmax / temp=0) to remove sampling overhead from timing
- Prefill timing includes KV cache construction — this is the correct measure of time-to-first-token cost

---

## Contributing

Results from different hardware are useful for building a reference dataset.
Open an issue with your JSON output and chip info to share your numbers.

---

## Related

- [llama.cpp](https://github.com/ggerganov/llama.cpp) — Inference of LLMs in C/C++
- [llama-bench](https://github.com/ggml-org/llama.cpp/tree/master/examples/llama-bench) — Native llama.cpp benchmark tool
- [MLX](https://github.com/ml-explore/mlx) — Apple's ML framework for Apple Silicon
- [mlx-lm](https://github.com/ml-explore/mlx-examples/tree/main/llms) — LLM inference + finetuning with MLX
- [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) — Whisper ASR with MLX
- [vLLM](https://github.com/vllm-project/vllm) — PagedAttention, continuous batching for production serving
- TNG Technology, [Prefill and Decode for Concurrent Requests (2025)](https://huggingface.co/blog/tngtech/llm-performance-prefill-decode-concurrent-requests)
