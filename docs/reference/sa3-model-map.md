# SA3 model map — what it was trained to do, how it works, where to hack it

Local doc (untracked). Distilled from the SA3 tech report ([paper PDF](../stable_audio_3_paper.pdf), arXiv 2605.17991) and checked against the PyTorch package. The upstream [model-overview.md](../guides/model-overview.md) is the friendly intro; this doc is the precise version plus the code map and the extension surface. Install, model lineup and community LoRAs live in [sa3-ecosystem.md](sa3-ecosystem.md).

**At a glance.** Two frozen-apart pieces. **SAME** (an autoencoder) turns 44.1 kHz stereo into a 256-dim latent at ~10.77 frames/s (4096× downsampling) and back. A **diffusion transformer (DiT)** generates those latents from three conditions: text (T5Gemma, cross-attention), duration (AdaLN *and* cross-attention), and an inpainting mask + masked reference latent (added per frame in every block). The DiT is trained in three stages (flow matching → distillation warmup → adversarial post-training) so the shipped model samples in 8 "ping-pong" steps with no CFG. Sequence length is variable: a 5 s request only computes ~120 frames, not the model's maximum.

## What it was trained to do (the capability envelope)

| Trained for | Detail |
|---|---|
| Text → instrumental music, SFX, single instruments/stems | One joint model for `medium`/`large`. `small` is split into `small-music` and `small-sfx` because SFX data hurt musical coherence at that size. |
| Variable length | small ≤ 2 min, medium/large ≤ 6 min 20 s (380 s). Cost scales with requested length. |
| Inpainting & continuation | **Jointly trained in every model, not a separate checkpoint.** Per training example: 80% no mask (plain generation), 10% 1–10 random masked spans, 10% causal mask (keep a prefix, generate the rest). |
| Clean endings | Trained with random silence extension (mean ~4 s) filled with the real "encoded silence" latent, so it learns to end on silence instead of at the sequence edge. |

**Not trained for:** vocals/speech/lyrics, non-English prompts, instruction editing ("remove the drums"), audio-reference/style conditioning, time-varying controls (chords, beats, loudness curves). The paper explicitly leaves those to post-release finetunes (LoRA / ControlNet-style). Medium/large data = AudioSparx (806k licensed tracks with rich metadata) + Freesound CC0/CC-BY/CC-Sampling+ (~472k, music filtered out with a PANNs tagger + copyright check).

**Prompt distribution it saw** (why prompts behave as they do): AudioSparx prompts are a random subset of metadata fields (description, title, keywords, BPM, genre, moods, instruments) joined by commas, with the field name prepended half the time (`Instruments: Guitar, Drums, Moods: Uplifting`), fields shuffled/subsampled half the time, lowercased half the time. Freesound prompts are half LLM-rewritten captions, half title + shuffled tags + description. So `Key: Value,` metadata-style prompts are in-distribution, and order/case shouldn't matter much. The paper says the `TrackType: Music, VocalType: Instrumental,` / `TrackType: SFX,` prefixes "significantly improve generation quality" (prompt advice itself: [prompting.md](../guides/prompting.md)).

**Known length biases (from the eval tables):** quality peaks at 120–190 s for music. At 20 s it drops (short training clips are mostly loops, not songs). At 380 s prompt adherence drops sharply (CLAP 0.39 → 0.28) because long training tracks are mostly ambient/classical, so long durations pull toward ambient/classical regardless of prompt. For SFX, longer → better FAD but worse prompt match.

## How it works

**SAME autoencoder** (detail: SAME paper, arXiv 2605.18613). Patchify stereo into 256-sample patches (256×), then a **Transformer Resampling Block** does 16× more: it interleaves a learnable "output" token after every group of 16 inputs, runs a transformer (differential attention + RoPE) over the whole thing, and keeps only the output tokens. The decoder does the reverse (several output tokens per input). Bottleneck = soft-normalisation (learnable affine with running std) → **deterministic** encoding, not a VAE sample. Losses: multi-resolution STFT (7 FFT sizes 32–2048, K-weighted, mid/side and L/R, with a phase loss), relativistic GAN, a small diffusion model trained jointly on the latent ("diffusion alignment", gradients reach the encoder), 1×1-conv linear regressors for **chroma** and **inter-aural level difference**, and a contrastive critic over (latent, audio features, T5Gemma text) triplets. Consequence: chroma and stereo image are *linearly decodable* from the latent by construction, and the latent is text-aligned.

**DiT block order** (per block, ×D): RMSNorm → AdaLN scale/shift → self-attention (QK-RMSNorm, RoPE on only 32 dims per head) → AdaLN gate → cross-attention to 257 tokens (256 T5Gemma tokens + 1 duration token, no positions) → **local-additive conditioning** (inpaint signal: 257-ch = 256 masked latent + 1 mask, through a zero-initialised 2-layer MLP, added to the residual) → RMSNorm → AdaLN → SwiGLU FFN (4d) → gate. Timestep + duration are Fourier-featured, summed, and turned into one shared AdaLN embedding plus a learned per-block bias ("AdaLN-Single"). 64 learned **memory tokens** (registers) are prepended to the sequence and dropped at the output. medium/large use differential attention (two attention maps subtracted), small does not.

| | d | blocks | heads | diff-attn | codec | DiT params |
|---|---|---|---|---|---|---|
| small | 1024 | 20 | 16 | — | SAME-S | 459M |
| medium | 1536 | 24 | 24 | ✓ | SAME-L | 1.4B |
| large | 2048 | 26 | 32 | ✓ | SAME-L | 2.7B (not released) |

**Training stages.**
1. **Flow matching** (`x_t = (1−t)x0 + tε`, predict `v = ε − x0`). Minibatch optimal-transport noise pairing (Sinkhorn), truncated logit-normal timesteps (t < 0.075 dropped), a **per-example timestep shift** that pushes long sequences toward higher noise (μ from 0.5 short → 1.15 long), loss split into separately-averaged "generated region" and "kept context" terms, padding masked out of attention and loss. CFG dropout 10% on text+timing. This is the `-base` checkpoint.
2. **Distillation warmup** (10k steps). A frozen teacher runs 15 DPM++ steps **at CFG 5**; the student learns to jump from any point on that trajectory straight to its endpoint (MSE). This is where CFG gets baked in. Output is blurry (regression to the mean).
3. **Adversarial post-training.** Teacher discarded. Generator = the warmed-up model, still velocity-parameterised (`x̂0 = x_t − t·v`). Discriminator = a copy of the *base* model (fully conditioned on text, duration and inpaint inputs), features from **layer 14** → small conv head → per-frame realness. Real and fake are re-noised to a shared random `t_D` with the same noise, and compared relativistically in prompt-matched pairs. The discriminator also gets a contrastive loss (real audio with shuffled prompts = negatives). The generator gets a **CLAP loss computed directly on SAME latents** (they trained a latent-space CLAP to avoid decoding). Optimiser: Muon on QKV/FFN projections + AdamW elsewhere; EMA 0.9995 on the generator.

**Inference.** Allocate `L = ⌈(d + 6 s)·44100/4096⌉` frames (6 s silence headroom, trimmed afterwards). Ping-pong sampling: predict `x̂0`, re-noise with *fresh* noise to the next lower t, repeat. 8 steps uniform in logSNR over [−6.2, 2.0], the same schedule at every length. That deliberately mismatches training's length shift; the paper reports it works. 1 step works but is much worse (music FAD 0.107 → 0.258 on medium).

## Where the code lives

| Paper concept | Code |
|---|---|
| `generate()` entry, prompt/duration → conditioning, 6 s headroom, inpaint mask from seconds | `stable_audio_3/model.py` (`StableAudioModel.generate`, `_adapt_sample_size`, `_encode_audio_input`) |
| Schedule + samplers (pingpong, euler, rk4, dpmpp), CFG/APG mixing, per-step `callback` | `stable_audio_3/inference/sampling.py` (`build_schedule`, `sample_flow_pingpong`, `sample_diffusion`) |
| LogSNR schedule and the length-shift families | `stable_audio_3/inference/distribution_shift.py` (`LogSNRShift`, default `rate=0` = length-invariant) |
| DiT wrapper: timestep features, prepend/memory, APG projection, `exit_layer_ix` early exit | `stable_audio_3/models/dit.py` |
| Transformer blocks, memory tokens, local-additive + **modular local** conditioning | `stable_audio_3/models/transformer.py` |
| Routing condition tensors to cross-attn / global / local-add / modular-local / prepend | `stable_audio_3/models/diffusion.py` (`ConditionedDiffusionModelWrapper.get_conditioning_inputs`) |
| Training mask sampler (random spans / causal) | `stable_audio_3/models/inpainting.py` |
| Flow-matching training loop (used for LoRA) | `stable_audio_3/training/diffusion.py` |
| SAME encoder/decoder | `stable_audio_3/models/autoencoders.py`; MLX ports `optimized/mlx/models/defs/same_{s,l}_decoder.py` |
| MLX DiT (what `./sa3` runs) | `optimized/mlx/models/defs/dit_mlx.py`, `dit_mlx_medium.py` |

Not in the repo: the distillation-warmup and adversarial stages, the discriminator, the latent-space CLAP and the SAME training losses. Only flow-matching (LoRA) training ships.

## Hacking / extension surface

Ordered by cost. "Hook" = what already exists in code; ideas marked *(untested)* are inferences from the paper, not results.

**No training (inference-time):**
- **Inpainting as a general editing primitive.** Any mix of keep/generate frames works (it was trained on 1–10 spans + causal). Use it for variations of a bar, loop extension, fixing a glitch, bridging two clips (keep both ends, generate the middle). Hook: `inpaint_audio` + `inpaint_mask` / `_start_seconds` / `_end_seconds` (MLX: `--inpaint-range`).
- **SDEdit / style re-render.** `init_audio` + `init_noise_level < 1` noises real audio and denoises under a new prompt.
- **Latent arithmetic on SAME.** Latents are deterministic, frame-aligned and semantically structured, so averaging, crossfading or interpolating latents is meaningful. Chroma and ILD are linearly readable *(untested: so a linear probe could also steer key/stereo width)*.
- **Prompt-embedding surgery.** Pass precomputed `conditioning_tensors` to interpolate between T5Gemma embeddings, or schedule different prompts across sampling steps via the sampler loop.
- **Duration as a style knob.** Duration conditions both AdaLN and cross-attention, and long durations bias toward ambient/classical. *(untested: decouple the `seconds_total` you condition on from the length you allocate.)*
- **Activation steering.** Community steering vectors already exist (`matteospanio/aria`, see ecosystem doc). Hook: `exit_layer_ix` for reading features. Block 14 is what the discriminator found semantically rich.
- **Guidance knobs.** The post-trained model expects `cfg_scale=1`. CFG > 1 doubles cost and tends to "crunch". The `medium-base`:`medium` merge from the ecosystem doc is the known way to tolerate higher CFG. APG (`apg_scale`) and negative prompts are wired in. `LogSNRShift(rate>0)` gives a length-dependent inference schedule *(untested; the paper only tried rate 0)*.
- **Per-step callback** gets `denoised` (x̂0) every step: use it for previews, early stopping, or guiding with an external scorer.
- **Where sampler tricks hook in the MLX runtime.** `optimized/mlx/scripts/sa3_mlx.py` builds a `model_fn(x, t)` closure (CFG/APG live there) and hands it to `sample_flow_pingpong` in `optimized/mlx/models/defs/sa3_pipeline.py`. Any per-step trick (prompt switching, regional prompts, cross-model or skip-layer guidance, FlowEdit-style editing) is a custom `model_fn` in our own script that imports these, never an edit to the upstream file. The flow-matching `-base` DiTs also exist as MLX weights (`dit_<model>-base_f16.npz` in `scripts/weights.py`, fetched for LoRA training), so inversion-style tricks that need `-base` don't need PyTorch. Ranked idea backlog with sources: the *Idea backlog* tab of the research Claude Doc (see [INDEX](../INDEX.md)).

**Light training:**
- **LoRA / DoRA** on the `-base` checkpoint, applied to the post-trained one (mechanics: [lora.md](../workflows/lora.md)). Since inpainting is part of every training step, LoRAs keep inpainting if the mask sampler stays on.
- **Decoder-only finetunes** of SAME (e.g. the community squeak-fix LoRA) are independent of the DiT.

**New conditioning (the designed extension point):**
- **`modular_local_cond_configs`** in the model config adds named, frame-aligned conditioning signals. Each gets its own zero-initialised MLP and is added per block exactly like the inpaint signal, so a pretrained checkpoint is unchanged until trained. This is the slot for ControlNet-style time-varying controls: chroma/chords, beat/onset grid, loudness envelope, stem latents, a "listen to this other track" latent. Shape `(B, channels, T_latent)` at ~10.77 Hz. The paper's own design note says the zero-init exists precisely to bolt new pathways onto pretrained models. No released checkpoint uses it.
- **New cross-attention tokens** (e.g. a CLAP/MERT audio-style embedding appended next to the 257 text+duration tokens) are possible but change the context length the attention was trained on. Riskier than modular-local.

**Expensive / research:** reproducing the adversarial post-training after a finetune (needs discriminator + latent CLAP, not released). Training on a different codec is off the table without retraining from scratch: the DiT is welded to SAME's latent space and frame rate.

## Invariants and traps

- **Upstream `model-overview.md` disagrees with the paper.** It lists small at 433M (paper: 459M), SAME-S 266M / SAME-L 1.7B (paper: 108M / 852M, likely learnable-params-only vs total), and medium max length ~4.75 min (paper and ecosystem doc: 380 s). Trust the paper for architecture and the HF config for runtime limits.
- **Post-trained checkpoints are not flow-matching models any more.** They predict x̂0 via a velocity head. Use ping-pong at ~8 steps and CFG 1. Running Euler at 50 steps or high CFG on them is off-distribution. The `-base` checkpoints are the opposite: true flow models trained for ~50 Euler steps at CFG ~7.
- **Inpaint mask polarity:** 1 = keep, 0 = generate. Masked latent = encoded audio × mask, concatenated with the mask (257 channels).
- **Silence is a real latent, not zeros.** Padding/headroom uses the encoding of a zero waveform. Zero-filling latents by hand is off-distribution.
- **Length ≠ free.** Cost and VRAM grow with length (medium: 5.5 GB at 30 s, 6.5 GB at 120 s+), and prompt adherence falls at max length. Generate at the length you need.
- **small + SAME-S is the weak link for edits**: rougher inpaint seams (FAD-full ≫ FAD-inpaint) than medium/large.

## Numbers worth remembering

Music (Song Describer, 120 s, 8 steps): medium FAD 0.107 / CLAP 0.390 / MOS-musicality 4.15, beating ACE-Step 1.5 (0.193) and DiffRhythm 2 (0.293), on par with closed Stable Audio 2.5. SFX (BBC, 5 s): medium FAD 0.369 vs Woosh Flow 0.580 and SAO 0.501. H200 latency for 120 s: small 0.45 s, medium 0.78 s (TensorRT 0.13 s). M4 MacBook Pro, small at 120 s: 5.9 s CPU-only, 3.1 s CoreML.
