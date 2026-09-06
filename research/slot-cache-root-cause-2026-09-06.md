# Slot-cache greedy gate — root cause (2026-09-06)

**Status:** root cause identified and reproduced offline; parity design proven offline; production fix not yet shipped.
**Baseline:** V1 Monolithic (33.8 tok/s C1). **Candidate:** slot cache S=112 (43.4 / 93.9 / 93.0 C1/C4/C8) — greedy 2/20 vs V1.

## What the divergence is

Two independent sources, found by teacher-forcing one divergent prompt through both servers with a forward hook on every
sub-module (615 hooks / 78 layers) and comparing tensors in forward order.

| source | evidence | scope |
|---|---|---|
| **A. torch.compile / Inductor** | V1 compiled vs V1 eager (`-O0`): 150/150 traced modules differ, first at `L03.mlp.gate`; greedy 2/20 between them | affects *every* build; not a cache defect |
| **B. bf16 routing-weight rounding** | V1 eager vs sc9 eager: prefill 0/615 identical; decode step 3 of prompt 12 first differs at **`L15.mlp.experts`** with identical inputs, ids and shared-expert output; offline on 3,041 real layer-3 tokens, NoAuxTc→pack→routed kernel mismatches Monolithic on **10/3041**, every one a single bf16 ulp on one weight sitting within 0.001 ulp of the rounding midpoint (frac 0.4990–0.5004) | Modular-class path only |

Mechanism of B: Monolithic routes in-kernel (`trtllm_fused_moe_routing_deepseek.cu`) and converts the fp32 weight to bf16 with
`OutputT{s * scale / sum}`. The Modular path uses `NoAuxTc` (`noAuxTcKernels.cu`), same arithmetic in source, but compiled in a
*different FlashInfer JIT module*. Both get `-use_fast_math`; the `tanhf` inside `sigmoid_accurate` lowers to different bits per
translation unit (~1e-6 relative). Only tokens whose weight lands on a bf16 midpoint see it. One ulp at layer 15 → different
argmax somewhere in the next 200 tokens on ~1 prompt in 10.

## What was falsified today (each with the probe that killed it)

| hypothesis | probe | result |
|---|---|---|
| attention / KV differs | `tf_decode.sh` decode trace, M=1 | L0–L14 all sub-modules bit-identical |
| LRU / cache contents wrong | `lru_unit.py`, live L15 experts input | 0 bookkeeping errors; expert input identical |
| routed kernel arithmetic differs | `monoweights.py` | Monolithic's own (ids, weights) → routed kernel = **0/3041** |
| fp32 unpacked weights fix it | `unpacked_eager.sh` (sc10) | **1/20**; even token 1 moved |
| emulate Monolithic's arithmetic in Python | `weightemul.py`, 24 variants (2 sigmoid × 4 sum orders × 3 divisions) | best 7/10 missed; not emulatable without the kernel's `tanhf` |
| use Monolithic `do_finalize=False` as router | `monoweights.py` | exact, but **297 µs** (vs NoAuxTc 4 µs) |
| use Monolithic with 128-wide dummy experts as router | `minimono.py` | exact (0/3041 ids+weights), still **293 µs** — launcher fixed overhead, not GEMMs |

## Parity designs that work (offline, 3,041 real tokens)

1. **Monolithic over sorted slots** (`slotmono.py`): remap logits/bias/scalars into slot space (empties −1e4), keep slot rows
   sorted by expert id so the in-kernel tie-break matches, call `trtllm_fp4_block_scale_moe(num_experts=S)`.
   **0/3041 at M=1, 0/1520 at M=2.** Cost: slot compute 203.5 µs vs 204.4 µs on 256 (same). Downside: sorted insert = whole-table
   permutation per miss (~0.35 ms/layer) and a double-buffered table (2× HBM → S must halve to 56). Draft hook: `probes/slot_cache_mono_v8_draft.py`.
2. **Routing FFI in Monolithic's JIT module** (not built): expose `Routing::Runner::run` standalone (~40 lines in
   `trtllm_fused_moe_kernel_launcher.cu` + binding) returning `(ids, bf16 weights)`. Same TU → same `tanhf` bits → routed
   kernel path stays as is, S=112, v7 hook with a router swap. Upstreamable. Needs a vendored patch in the image.

## Gate implications

- Source A means "20/20 vs V1 compiled" is not a property any eager or differently-compiled build can have. The gate has to
  compare like with like (both eager, or both same compilation config) — or be defined on a quality metric rather than bit parity.
  This is James's call (contract A/B/C in the Astra round-2 notes).
- Source B is a real Modular-path defect and is fixable; it also affects upstream vLLM users of the Modular TRT-LLM NVFP4 class.

## Numbers to carry forward

- sc8 (S=112, TRT router, compiled): 43.4 / 93.9 / 93.0 tok/s C1/C4/C8. V1: 33.8 / 57.7 / 57.6. Eager: V1 6.9, sc9 5.5.
- Slot compute cost == full-expert compute cost (203 vs 204 µs) — the cache's entire gain is bytes not read over C2C.
- Live hit-rate instrumentation still missing (stats thread wrote 0 lines in every run) — fix before any S tuning.
