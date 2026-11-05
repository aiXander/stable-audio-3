# Docs index — where every doc in this repo lives

The one-screen map of **every** doc in this repo, so an agent can jump to the right file without grepping. Doc depth lives in the docs themselves; this file only says *what each one covers and when to open it*.

This repo is a clone of upstream `Stability-AI/stable-audio-3`, so docs come in two kinds:

- **upstream** — tracked by git, owned by Stability AI, replaced on every `git pull`. Read them; **don't edit them** (local edits conflict on pull). Record local corrections in a `local` doc instead.
- **local** — untracked files that are ours (`git status` shows them as `??`). Edit freely; this is where the `docs/` workflow (TODO → reference → finished) applies.

## Start here — by task

| You're about to… | Open |
|---|---|
| Generate audio on this Mac (MLX `./sa3`), pick a model, or debug the local install | [reference/sa3-ecosystem.md](reference/sa3-ecosystem.md), then [mlx/README.md](../optimized/mlx/README.md) |
| Look for community finetunes, LoRAs, merges or runtime ports | [reference/sa3-ecosystem.md](reference/sa3-ecosystem.md) (*Community ecosystem*) |
| Understand the architecture, what SA3 was trained to do, or plan a hack / extension (new conditioning, editing tricks, steering) | [reference/sa3-model-map.md](reference/sa3-model-map.md) (paper distilled + code map); gentler intro in [guides/model-overview.md](guides/model-overview.md) |
| Write or tune prompts | [guides/prompting.md](guides/prompting.md) |
| Use the PyTorch package (`stable_audio_3/`): text-to-audio, audio-to-audio, inpainting/continuation, chunked decoding | [workflows/inference.md](workflows/inference.md) |
| Encode/decode audio with SAME directly, or pre-encode a dataset | [workflows/autoencoder.md](workflows/autoencoder.md) |
| Train a LoRA on Apple Silicon (MLX) | [mlx/README.md § LoRA training](../optimized/mlx/README.md#lora-training), [mlx/TRAINING_CONVENTIONS.md](../optimized/mlx/TRAINING_CONVENTIONS.md) |
| Train a LoRA with PyTorch, or learn the adapter types (LoRA / DoRA / BoRA / -XS) | [workflows/lora.md](workflows/lora.md) |
| Choose a runtime for some hardware (MLX / TFLite / TensorRT) | [optimized/README.md](../optimized/README.md) |

## Full inventory

### `docs/` — this folder

| Doc | Kind | What it covers |
|---|---|---|
| [reference/sa3-ecosystem.md](reference/sa3-ecosystem.md) | local | Official model lineup, the community repos worth trying, how *this* Mac install is wired (MLX wrapper, weights, output paths), its traps, and the HF survey refresh script. |
| [reference/sa3-model-map.md](reference/sa3-model-map.md) | local | The paper distilled: capability envelope (what it was / wasn't trained for, prompt distribution, length biases), SAME + DiT internals, the 3 training stages, inference, paper-concept → code map, the hacking/extension surface, and where upstream docs disagree with the paper. |
| [guides/model-overview.md](guides/model-overview.md) | upstream | Architecture: SAME codec, diffusion transformer, provided checkpoints, how inference works, training data. Uses `training-stages.png`. |
| [guides/prompting.md](guides/prompting.md) | upstream | Prompt guide per mode (music, stems/solo instruments, samples/SFX, init audio/inpainting), with example prompts and AudioSparx tags. |
| [workflows/inference.md](workflows/inference.md) | upstream | PyTorch inference API and controls per mode, per-batch overrides, chunked decoding, LoRA at inference time. |
| [workflows/autoencoder.md](workflows/autoencoder.md) | upstream | SAME encode/decode API, chunking long audio, saving latents, `scripts/pre_encode_dataset.py`. |
| [workflows/lora.md](workflows/lora.md) | upstream | PyTorch LoRA training (`scripts/train_lora.py`): CLI, layer filtering, memory, resuming, adapter types, multi-LoRA merging, Gradio LoRA controls. |
| [stable_audio_3_paper.pdf](stable_audio_3_paper.pdf) | local | The SA3 technical report (arXiv 2605.17991). Primary source for architecture and training claims. |
| `guides/*.png` | upstream | Images used by the upstream guides and README. |

### `optimized/` — runtime docs (all upstream)

Each runtime is self-contained: its own installer, `sa3` CLI, `sa3-gradio` UI and README.

| Doc | What it covers |
|---|---|
| [optimized/README.md](../optimized/README.md) | Picker table: which runtime for which hardware, with one-liners. |
| [mlx/README.md](../optimized/mlx/README.md) | **The runtime we run.** Apple-Silicon MLX: install, models and modes, flag reference, speed/memory, LoRA training, file map. |
| [mlx/TRAINING_CONVENTIONS.md](../optimized/mlx/TRAINING_CONVENTIONS.md) | Every training default of `underfit` (the reference SA3 LoRA trainer) and how the MLX trainer matches it: adapter config, checkpoint format, loss, timesteps, CFG dropout, optimizer, data pipeline, parity findings. Open before changing MLX training code. |
| [mlx/ableton/README.md](../optimized/mlx/ableton/README.md) | Experimental: insert `sa3` generations into Ableton Live at the playhead. |
| [tflite/README.md](../optimized/tflite/README.md) | Portable CPU runtime (LiteRT/TFLite; macOS/Linux/Windows), precision variants, flags. |
| [tflite/docs/RUNGS.md](../optimized/tflite/docs/RUNGS.md) | Why the TFLite SAME models are a ladder of fixed-shape "rung" subgraphs (dynamic shapes blow up RAM), tiling, fp32 vs w8a8, the baked-in output limiter. |
| [tflite/build/README.md](../optimized/tflite/build/README.md) | Rebuilding every TFLite model from the PyTorch code + checkpoints. |
| [tensorRT/README.md](../optimized/tensorRT/README.md) | NVIDIA runtime: whole pipeline as one CUDA graph, install, flags, speed. |
| [tensorRT/build/README.md](../optimized/tensorRT/build/README.md) | Building TRT engines for a GPU arch other than Hopper (`sm_90`), fp8 recipes, publishing engines. |
| [tensorRT/quantize/README.md](../optimized/tensorRT/quantize/README.md) | Quantized SAME encoder/decoder tiers (GPTQ, per-op precision) and which to pick. |
| [tensorRT/benchmarks/README.md](../optimized/tensorRT/benchmarks/README.md) | SAME autoencoder benchmarks across lengths/precisions, and two measurement traps. |
| [tensorRT/scripts/lora/README.md](../optimized/tensorRT/scripts/lora/README.md) | Using runtime-swappable LoRA/DoRA on the TRT DiTs. |
| [tensorRT/scripts/lora/BUILDING_TRT_LORA.md](../optimized/tensorRT/scripts/lora/BUILDING_TRT_LORA.md) | Deep dive on how live LoRA was built into the TRT engines, and why branches beat refit. |

### Outside the repo

| Where | What |
|---|---|
| [Root README](../README.md) (upstream) | Models table, install, quick start, CLI, hardware support, testing. |
| SAME-L research notes — Claude Doc `https://claude.ai/artifact/DypQNkdfui1FG9pJosuHu7` | Xander's running notes on SAME-L / SA3: links and papers (with read status), ideas, open questions, dated experiment log. Its *Idea backlog* tab is a ranked list of ~50 sourced tricks (sampler hacks, codec bending, light training, models to chain) — start there when choosing the next experiment. New links, ideas and experiment results go there, not into this repo. |
| `AI_Audio/outputs/` (parent folder) | Where we keep generated audio. |

## Keeping this index current

- Add a row whenever you create, move or delete a doc (local or upstream after a `git pull`). One row per doc; never copy doc content here.
- `docs/TODO/` (plans) and `docs/finished/` (retired plans) don't exist yet — create them on first write. List the `TODO/` folder here, not individual plans; never list `finished/` files.
