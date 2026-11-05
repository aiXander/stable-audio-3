# Stable Audio 3 — our fork

## Mission

Xander's fork (`aiXander/stable-audio-3`, a submodule of `AI_Audio`) of upstream `Stability-AI/stable-audio-3` (open text-to-audio / music / SFX diffusion models plus the SAME audio codec). Xander uses it to generate audio on this M2 Mac through the Apple-Silicon MLX runtime (`optimized/mlx/`) and to research the SAME-L autoencoder and SA3 finetuning. Nothing here is deployed; generated audio goes to `AI_Audio/outputs/` in the parent folder.

## Blast radius

- **`origin` is Xander's fork; `upstream` is Stability AI.** Code edits are welcome: commit on `main`, push to `origin`, then commit the new submodule pointer in the parent `AI_Audio` repo. Never push or open PRs to `upstream` unless Xander asks. Pull Stability's updates with `git fetch upstream && git merge upstream/main`.
- **Our docs are committed to the fork** (`CLAUDE.md`, `AGENTS.md`, `docs/INDEX.md`, `docs/reference/`). The paper PDF is local only (`.git/info/exclude`).
- **Disk is nearly full.** Ask before downloading large weights (Medium + SAME-L ≈ 6 GB) or building datasets; check `df -h ~` first.
- **Free to run without asking:** `./sa3` generations, the gradio UI locally, `uv run pytest`, reading HF model cards and the HF API.

## Documentation map

**[docs/INDEX.md](docs/INDEX.md) is the map of every doc in the repo** — upstream guides, per-runtime READMEs under `optimized/`, our local reference docs and the external research notes — with a "start here by task" table. Open it whenever you need docs on any topic instead of grepping for READMEs. Keep it current: add a row when you create, move or delete a doc.

| Local reference doc | Open when |
|---|---|
| [docs/reference/sa3-model-map.md](docs/reference/sa3-model-map.md) | you need to know how SA3 works or what it was trained for (from the paper), where a paper concept lives in code, or you're planning an extension (new conditioning, inpainting tricks, steering, finetunes) |
| [docs/reference/sa3-ecosystem.md](docs/reference/sa3-ecosystem.md) | running or debugging the local MLX install, choosing a model, or looking for community LoRAs / finetunes / ports |

Upstream docs (`docs/guides/`, `docs/workflows/`, `optimized/**/README.md`): leave them as upstream wrote them so merges stay clean; corrections go in a local reference doc. SAME-L research links, ideas and experiment results go in Xander's Claude Doc (URL in `docs/INDEX.md`), not in the repo.

## Repo layout

- `optimized/mlx/` — the runtime we use (pure MLX, own `.venv/`, `./sa3` wrapper). Trap: relative `--out` paths resolve under `optimized/mlx/output/`, not your cwd.
- `optimized/tflite/`, `optimized/tensorRT/` — CPU and NVIDIA runtimes; not used on this machine.
- `stable_audio_3/` — the upstream PyTorch package (inference, models, LoRA training); `scripts/` holds its LoRA training and dataset pre-encode entry points.
- `tests/` — pytest suite for both the PyTorch package and the MLX training code.

## Load-bearing rules

1. Edit upstream code when an experiment needs it, but keep each change small and in its own commit with a clear message: every `git merge upstream/main` has to reconcile them.
2. Don't run `optimized/mlx/bootstrap.sh` from an agent shell: it ends by `exec`-ing an interactive shell and hangs. Use `install.sh -y` and `./sa3` directly.
3. MLX model names differ from upstream IDs (`sm-music` vs `small-music`); check which stack a doc is about before copying a flag.

## Definition of done

- Generated audio lands where Xander expects (absolute `--out` into `AI_Audio/outputs/` unless told otherwise) and you name the file in your summary.
- Doc pass done: new findings about the install or ecosystem are in `docs/reference/`, and `docs/INDEX.md` lists every doc that exists.

## Commands

- Generate (MLX): `optimized/mlx/sa3 --dit {sm-music,sm-sfx,medium} --decoder {same-s,same-l} …` — `--help` for the rest; missing weights auto-download on first use.
- Reinstall / update the MLX env: `optimized/mlx/install.sh -y` (after `git pull --ff-only`).
- Tests: `uv sync --group dev && uv run pytest`.
