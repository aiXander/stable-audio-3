# Stable Audio 3 — model landscape & local install

Local notes, **not part of the upstream Stability-AI repo** (this file is untracked; `git pull` leaves it alone). Two jobs: (1) which SA3 checkpoints, finetunes, LoRAs and ports exist and which are worth trying; (2) how *this* Mac install is set up and where it bites. Architecture, training, sampling and the extension surface: [sa3-model-map.md](sa3-model-map.md); LoRA training mechanics: [lora.md](../workflows/lora.md).

Landscape surveyed **2026-11-05** (HF model tree + web). It moves fast — re-run the refresh in the runbook before trusting "nothing exists yet" claims.

## Where the code lives (local install)

- `optimized/mlx/` — the Apple-Silicon (MLX) stack we actually run. Installed via `install.sh -y`; project-local `.venv/` (uv, Python 3.11). Bundles on disk: `sm-music` + `medium` (DiT + SAME-L encoder/decoder). The gradio extras (`requirements-gradio.txt`) are installed in the same venv.
- `optimized/mlx/sa3` — CLI wrapper (handles the venv). `--dit {sm-music,sm-sfx,medium}`, `--decoder {same-s,same-l}`.
- `optimized/mlx/scripts/weights.py` — weights manifest; pulls from HF repo `stabilityai/stable-audio-3-optimized` into the HF cache and symlinks into `models/mlx/`. On this Mac the HF cache for SA3 is `AI_Audio/models/hf-cache` (set via `HF_HUB_CACHE` by `sa3lab/sa3`, `sa3lab/studio` and the project's `.claude/settings.json`); a bare upstream `./sa3` without that env var would download missing weights into `~/.cache` instead. Missing weights auto-download on first `./sa3` use.
- `optimized/mlx/scripts/lora_train_mlx.py` — local LoRA training (MLX); the optimized repo also ships the **base** DiT weights needed for it.
- `optimized/mlx/sa3-gradio` → `scripts/sa3_gradio.py` — the local web UI (all modes: t2a, a2a, inpainting, LoRA slots, history). We don't run it directly: `AI_Audio/sa3lab/studio` launches it re-skinned, localhost-only, with a full-HD spectrogram.
- Outputs: we keep generated audio in `AI_Audio/outputs/` (outside the repo). `optimized/mlx/output/gradio` is a git-ignored **symlink** to `AI_Audio/outputs/sa3-gradio/`, so UI renders land there too (MP3 by default; WAV under the UI's Output options).

## Official lineup

| Model | Params | Max len | Open? | MLX `--dit` | Decoder |
|---|---|---|---|---|---|
| Small-Music | 433M | 120 s | ✅ | `sm-music` | `same-s` |
| Small-SFX | 433M | 120 s | ✅ | `sm-sfx` | `same-s` |
| Medium (music + SFX) | 1.4B | 380 s | ✅ | `medium` | `same-l` |
| Large | 2.7B | 380 s | ❌ API / enterprise only | — | — |

- Each open model also has a **`-base`** checkpoint (pre post-training) — the intended starting point for LoRA/finetunes. Post-trained = better out of the box; base = better to train on.
- Codecs **SAME-S** / **SAME-L** are published standalone (`stabilityai/SAME-S`, `stabilityai/SAME-L`). The two small DiTs share bit-identical SAME-S weights.
- License: Stability AI Community License (commercial use allowed under its revenue threshold). Derivatives (LoRAs, merges) inherit it.
- No SA3.1 / newer release and no open Large as of the survey.

## Community ecosystem — what's actually useful

Most of the ~75 "stable-audio-3" HF repos are mirrors or format re-packs of the official weights. The ones that add something:

| HF repo | Type | Why it matters |
|---|---|---|
| `worstplayer/Stable_Audio_3_medium_base_merge-GGUF` | merge | Medium-base : Medium at **1:2** (author's example: 16 steps, CFG 3) lets you raise CFG without collapse; author reports it largely removes the "**SA3 crunch**". GGUF for `sa3.cpp`. The merge recipe itself is trivially reproducible on MLX weights: the base and ARC npz have identical keys (525), `AI_Audio/sa3lab/bench/quality_bench.py` builds it on demand (needs the 2.9 GB `dit_medium-base_f16.npz`). **Our blind A/B on 2026-11-05 found no gain over plain medium**, with or without CFG 3, so neither file is kept on disk. |
| `thepatch/same-l-decoder-lora` (`squeakfix_v3`) | decoder LoRA | Fixes hi-freq percussion squeaks that **compound** when audio round-trips encode/decode (continuations, audio-to-audio chains). Decoder-only, so stacks with any DiT LoRA. Officially needs a `decoder-lora` PyTorch branch, but on MLX it merges straight into the SAME-L decoder weights (plain LoRA, scale 1.0; PyTorch names map 1:1 to the MLX npz, e.g. `layers.3.transformers.N.ff.ff.0.proj` → `blocks.N.ff.glu_proj`): see `squeakfix_deltas()` in `AI_Audio/sa3lab/bench/quality_bench.py`. |
| `motiftechnologies/stable-audio-3-maqam-lora` | style LoRA | Arabic Maqam on Medium; rank 64, 100 expert-labelled tracks, one H100. The only real genre LoRA found — a useful reference recipe. |
| `matteospanio/aria` | steering vectors | Activation-steering directions (taste → "sonic seasoning") for the `aria` C/CUDA runtime. Shows SA3 is steerable without training. |
| `scragnog/HOT-Step-CPP-StableStep` | pipeline use | Medium as an SDEdit-style **post-processor**: re-renders another model's instrumental to replace codec fizz, vocals untouched. |
| `shuoyang-zheng/stable-audio-3-tilde` | port | SA3 inside **Max/MSP** (`neural.gen~`). |
| `petalsfm/foundry` | app | macOS 16-pad drum machine, each pad = Small-SFX prompt, offline. |
| `shoegazerstella/stable-audio-morph-onnx` | port | SAME-S encoder/decoder in ONNX for browser latent morphing. |

Runtime ports (pick by target, not quality — they're the same weights): GGUF / `sa3.cpp` (`thepatch/*-GGUF`), MLX 4/8-bit (`aufklarer/*`, `MusicForge/*`, `mlx-community/*`), ONNX, CoreML/iOS (`arraypress/*`), Qualcomm ExecuTorch. **ComfyUI** has native support (`Comfy-Org/stable-audio-3`, the most-downloaded SA3 repo).

Hosted training: fal.ai `stable-audio-3-trainer` (zip of clips + captions; trains on medium-base / small-music-base / small-sfx-base).

## Invariants and traps

- **`./sa3 --out` relative paths ignore your cwd**: the wrapper `cd`s into `optimized/mlx/` and `sa3_mlx.py` prefixes any relative path with `optimized/mlx/output/` (no `--out` → auto-named file there). So `../../../outputs/x.wav` lands in `stable-audio-3/outputs/`. Pass an absolute `--out` to put files anywhere else.
- **Disk.** The disk is >90% full. The full MLX weight set (all DiTs, base DiTs, both codecs) is ~13 GB; `medium` + SAME-L is ~6.3 GB (already pulled); each `-base` DiT is 0.9 GB (small) or 2.9 GB (medium); LoRA training adds datasets + checkpoints. Check `df -h ~` before pulling more.
- **Upstream `./sa3-gradio` is exposed by default:** a public `gradio.live` share link, and it binds `0.0.0.0` (reachable from the LAN) even with `--no-share`. `sa3lab/studio` forces `127.0.0.1` and no share.
- **Upstream's spectrogram is low-res by construction** (fixed hop → ~16 columns/s, 30 mels, then upscaled), so short clips look blurry and the first column lights up white. `sa3lab/hispec.py` replaces it at runtime. Its CFG slider stops at 0, so negative CFG needs the `./sa3` CLI.
- **No HF token needed** for `stabilityai/stable-audio-3-optimized` (the MLX weights). Some community cards say the *upstream* PyTorch repos are gated — that only matters if we switch to the root `stable_audio_3/` PyTorch package.
- **Don't run `bootstrap.sh` from an agent shell**: its last step `exec`s an interactive shell on `/dev/tty` and hangs. Run `install.sh -y` + `./sa3` directly.
- **MLX bundle names ≠ upstream model IDs**: MLX uses `sm-music`/`sm-sfx`/`medium`; the PyTorch package and HF use `small-music`/`small-sfx`/`medium`.
- Measured on this Mac (base M2, 10-core GPU, 24 GB unified memory): `sm-music` + `same-s`, 30 s clip → 5.1 s wall (5.8× realtime), 1.7 GB peak RAM. `medium` + `same-l`, 20 s clip → 10.9 s wall incl. weight load (1.8× realtime; sampling ~0.6 s/step), 3.6 GB peak RAM. Realtime factor grows with clip length (upstream M4 Pro: ~29× at 120 s).

## Runbook

**Refresh the landscape survey** (finds every HF repo that declares an SA3 checkpoint as its base):

```python
import json, urllib.request
bases = ["stable-audio-3-medium", "stable-audio-3-medium-base", "stable-audio-3-small-music",
         "stable-audio-3-small-music-base", "stable-audio-3-small-sfx", "stable-audio-3-small-sfx-base",
         "SAME-L", "SAME-S"]
for b in bases:
    for rel in ["adapter", "finetune", "merge", "quantized"]:
        u = f"https://huggingface.co/api/models?filter=base_model:{rel}:stabilityai/{b}&limit=200&sort=downloads"
        for m in json.load(urllib.request.urlopen(u)):
            print(b, rel, m["id"], m.get("downloads"), m.get("createdAt", "")[:10])
```

Use `filter=` — the `other=` param works on the website URL but is silently ignored by the API (returns all models). The tree misses repos that don't tag a `base_model`, so also run `https://huggingface.co/api/models?search=stable-audio-3`. Then read model cards (`https://huggingface.co/<repo>/raw/main/README.md`) — most "finetunes" are re-packs; only list ones that change behaviour.

**Update the install:** `git -C stable-audio-3 pull --ff-only`, then `optimized/mlx/install.sh -y` (reuses `.venv/` if the Python version matches).
