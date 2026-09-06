# Slot cache greedy-gate investigation — 2026-09-05 (afternoon/evening)

**Status at end of day:** the slot cache serves the full 744B GLM-5.3 on one GB300 at **43.4 tok/s single-stream, 93.9 @4, 93.0 @8** (V1: 33.8 / 57.7 / 57.6) and adds **zero** numeric drift. It still fails the greedy gate (**2/20** vs V1) because any build that forces the `TrtLlmNvFp4ExpertsModular` class diverges from the Monolithic-class V1 — and after a full day of component-level ablation, **every component of the layer-3 MoE path is proven bit-exact**. The drift is somewhere else. Next step is a teacher-forced, per-layer diff (see `teacher-force-plan.md`).

Box: `milo@192.168.1.9`, `vllm-glm53-uva:v0.28.0-2cf0a691`, all runs in `/home/milo/big-v1-campaign/` (`LEDGER.md` is the authoritative timeline).

## Runs

| run | class | router | cache | greedy vs V1 | vs sc8 | C1 |
|---|---|---|---|---|---|---|
| V1 (`glm53-big-v1-keep`) | Monolithic | in-kernel | — (33 layers resident, offload 188) | ref | 2/20 | 33.8 |
| sc8-s112 | Modular | vLLM `grouped_topk` | S=112 | **2/20** | — | **43.4** (C4 93.9, C8 93.0) |
| ctl-modular | Modular | vLLM | built, bypassed every token | 2/20 | **20/20** | 21.0 |
| ctl-modular self-repeat (12 min apart) | | | | | 20/20 | |
| ctl-mono-420 | **Monolithic** | in-kernel | none; exact_pin; offload 420 | **20/20** | 2/20 | 23.6 |
| sc9-s112-trt | Modular | TRT `NoAuxTc` | S=112 (accidentally bypassed: `BYPASS=0` ⇒ bypass all) | **2/20** | **2/20** | (20.9, meaningless) |

Reading: cache innocent (sc8 ≡ ctl). Offload budget and exact pinning innocent (ctl-mono-420 ≡ V1). Router swap *moves* the divergence (sc9 ≠ sc8 ≠ V1, different first-diff positions per prompt) but does not close it.

## Offline ablations (layer 3, real weights, real captured tokens, Monolithic kernel as reference)

| test | result |
|---|---|
| routing 2×2: {vLLM router, TRT} × {packed bf16, unpacked fp32}, 14 decode tokens | packed variants **14/14 bit-exact**; unpacked 0/14 (Monolithic rounds routing weights to bf16 internally) |
| autotune tactic table: V1's (tuned) vs slot-cache key (miss → fallback) | 14/14, maxabs 0.0 |
| prefill M=512, all four {class} × {tactic table} | 512/512, maxabs 0.0 |
| batch composition (token alone vs in 512) | 0.0 |
| weight pipeline: stock `process_weights_after_loading` Monolithic vs Modular vs what the hook produced (w13, w2, both scale tables, g1c, g1a, g2a) | all 7 **bit-exact** |
| activation quant: both classes share `prepare() → _quantize_input → scaled_fp4_quant` from the same `quant_config`, before the class split | identical by construction (read, not measured) |
| **router on 3,041 decode tokens** (device ring buffer inside the graph): vLLM vs Monolithic replay | **3 expert-set mismatches — all exact fp32 ties at the 8th expert** (23/3041 tokens tie) |
| TRT `NoAuxTc` vs Monolithic replay, same tokens | 0 mismatches |
| Monolithic routing M=1 vs M=512 (87 tokens incl. all ties) | 0 — batch-independent |
| vLLM vs NoAuxTc routing **weights** on identical ids | fp32 maxabs 1.5e-6 → **13/3041 differ after bf16 packing** |

So the vLLM router had two real defects relative to the kernel (tie-break, last-bit normalization); `NoAuxTc` fixes both and matches the kernel on ids and weights. And a server built on it is still 2/20.

## What this means

With weights, activation quant, ids, weights, packing, kernel, tactics, shapes, wrappers and the cache all exact, the layer-3 MoE forward cannot produce the drift. The remaining hypotheses are outside the MoE kernel call — things that change when `is_monolithic` flips at the runner level: shared-expert overlap/order (`SharedExpertsOrder`, multi-stream), in-place modification of `hidden_states` in the modular `select_experts` path, `mk_can_overlap_shared_experts`, spec-decode/MTP handling, or a per-layer effect that only shows on layers other than 3. None of these are testable from a layer-3 capture. The right instrument is a teacher-forced per-layer comparison: identical token sequence through V1 and sc9, hidden state recorded at every layer boundary, first differing (layer, position) wins.

## Fixes that came out of this day (all landed in `slot_cache_hook.py` / `exact_pin.py`)

- `exact_pin`: global keepalive list leaked every replaced block (9.0 GiB/layer vs 4.5) → per-storage `cudaFreeHost` weakref finalizer. sc8 loads at 428 GB host.
- Capture hooks must be graph-safe: never allocate or `torch.save` or sync inside CUDA graph capture (two launches died to this).
- `SLOT_CACHE_ROUTER=trt`: route Modular with `NoAuxTc(1,1)` — the only valid no-group config for 256 experts; `fused_topk_deepseek`'s Python wrapper rejects `(0,0)`.
- `SLOT_CACHE_LOGIT_RING=N`: device-side logits/ids ring for layer 3, dumped on a sentinel file — 3,041 tokens per greedy run instead of 14.
- `SLOT_CACHE_BYPASS_TOKENS=0` means *bypass everything*; the cache-on control needs the default (16).

## Corrections to earlier claims

- "Modular is 38% slower than Monolithic" — confounded (control read more bytes). Equal-bytes: Monolithic 23.6, Modular 21.0 (11%). sc8's 43.4 vs V1's 33.8 is roughly a matched-HBM-budget comparison (+28% single-stream).
- "bf16 packing is the top suspect" — backwards; packed bf16 *is* what the Monolithic kernel does.
- "prefill regime / tactic selection" — falsified.
- "exact ties are the root cause" — real defect, fixed, not sufficient.

Ablation and probe scripts: `hostleak_test.py`, `fix_pin3.py`, `ablate_routing.py`, `ablate_prefill.py`, `diff_autotune.py`, `weightdiff.py`, `weightdiff2.py`, `router3.py`, `router4.py`, `firstdiff.py` (this repo, `probes/`).
