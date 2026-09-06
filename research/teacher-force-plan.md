# Teacher-forced per-layer diff: V1 vs sc9 — plan (2026-09-05 evening)

## Question
For one prompt where V1 and sc9 diverge, at which **layer** and which **sub-module** does the residual stream first differ, given identical input tokens?

## Why this instrument
Greedy runs diverge and then *stay* diverged (KV pollution), so comparing outputs says nothing about where. Teacher-forcing feeds both servers the **same token sequence** (V1's greedy output for the prompt), so every position's input is identical by construction; the first differing hidden state is the cause, not a consequence.

## Design
- **Prompt:** greedy prompt 5 (V1 vs sc8 diverged at char 1; V1 vs sc9 at char 3 — divergence is immediate, so a short sequence suffices).
- **Sequence:** V1's greedy completion for prompt 5 (`runs/v1/greedy.json`), tokenized with the model tokenizer → `ids`. Feed **prompt + first 64 completion tokens** as a single prefill request with `max_tokens=1`, so every layer sees the identical M=len(ids) batch on both servers. This is teacher-forcing in one call — no per-token decode needed for localisation. (Decode-path check is a follow-up: if prefill matches but decode diverges, re-run as `max_tokens=1` requests of growing prefix length.)
- **Hook (generic, both servers):** `sitecustomize`-loaded module that, on `Glm4MoeDecoderLayer` (all 78) and inside each: `self_attn`, `post_attention_layernorm`, `mlp` (dense or MoE), `mlp.shared_experts` (when present), `mlp.experts` (RoutedExperts), registers **forward hooks** that record `(sum, sum_abs, sum_sq, first-64-elements-bf16-bits)` of the output on device into a preallocated buffer (graph-safe: fixed shapes, no allocation, no sync). Full tensors for layers 0–5 also saved (prefill at M≈100 is not graph-captured with piecewise cudagraphs above the capture sizes; safe to `torch.save` there, guarded by `is_current_stream_capturing`). Dump on a sentinel file, like the logit ring.
- **Comparison:** offline script pairs the two dumps by module name; reports first module (in forward order) where the stats differ, and the maxabs of the full tensor where available.

## Steps
1. `layer_trace.py` hook + `SLOT_CACHE_TRACE=1` (V1 gets `LAYER_TRACE=1` only — it must not load the slot-cache hook). Dry-run on the 6-layer synthetic model for graph safety.
2. `tf_prompt.py`: build the teacher-forced request from `runs/v1/greedy.json[5]` (prompt + 64 tokens), send to `:30001` with `max_tokens=1, temperature=0`, `logprobs=5`; also record the returned top-5 logprobs (a second, independent divergence signal at the output).
3. V1 is up (restored 19:48): land trace hook → **restart V1 with the trace env** (~20 min) → send request → dump → copy `trace_v1.pt`.
4. Restart sc9 with trace env (~25 min, cached autotune) → same request → `trace_sc9.pt`.
5. `tf_compare.py` → first differing (layer, module, position). Ledger + repo + blog.
6. Restore V1 (no trace env).

Box time ≈ 20 + 25 + 20 min ≈ 65 min; wall ≈ 1h15 with copies and comparison.

## Exit criteria
- First differing module identified with a bit-level statistic, on identical input tokens.
- If layer 0 already differs at `self_attn` → the class flip changes something in attention/graph mode, not MoE at all.
- If the first differing module is `mlp.experts` at layer k>3 → layer-3 ablations were representative but something is layer-specific (e.g. an expert count / scale edge case).
- If `mlp.shared_experts` or the residual add differs → shared-expert overlap order.

## Not doing
- No new kernel ablations. No blog edits until localised. No production swap: V1 goes back up at the end.
