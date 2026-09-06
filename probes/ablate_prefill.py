#!/usr/bin/env python3
"""ablate_prefill.py — same real layer-3 weights + real captured tokens, tiled to prefill-sized M.
Runs Monolithic and A (routed, packed bf16 = what sc8 runs) under whichever autotune table AT_CONFIGS points at,
logs which tactic resolved, dumps per-token outputs. Compare mode diffs dumps.
env: AT_CONFIGS=<json>  TAG=<name>  M_TARGET=512  |  CMP=tagA,tagB
"""
import os, glob, json, torch
import flashinfer
from flashinfer.fused_moe import RoutingMethodType
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import grouped_topk
from vllm.model_executor.layers.fused_moe.utils import trtllm_moe_pack_topk_ids_weights
from flashinfer.autotuner import AutoTuner

dev = "cuda"; E, K, HID, INTER = 256, 8, 6144, 2048
MT = int(os.environ.get("M_TARGET", "512"))
res = {"M_target": MT}
cmp = os.environ.get("CMP")
if cmp:
    a, b = cmp.split(",")
    for kind in ("mono", "A"):
        X = torch.load(f"/w/capture/pf_{kind}_{a}_M{MT}.pt"); Y = torch.load(f"/w/capture/pf_{kind}_{b}_M{MT}.pt")
        d = (X - Y).abs().max(dim=1).values
        res[f"{kind}:{a}_vs_{b}"] = dict(tokens=int(X.shape[0]), exact=int((d == 0).sum()), maxabs=float(d.max()))
    # cross-kind under each table
    for t in (a, b):
        X = torch.load(f"/w/capture/pf_mono_{t}_M{MT}.pt"); Y = torch.load(f"/w/capture/pf_A_{t}_M{MT}.pt")
        d = (X - Y).abs().max(dim=1).values
        res[f"mono_vs_A@{t}"] = dict(tokens=int(X.shape[0]), exact=int((d == 0).sum()), maxabs=float(d.max()))
    # and the decisive one: V1's real config (mono @ bigv1 table) vs sc8's real config (A @ slot table)
    X = torch.load(f"/w/capture/pf_mono_{a}_M{MT}.pt"); Y = torch.load(f"/w/capture/pf_A_{b}_M{MT}.pt")
    d = (X - Y).abs().max(dim=1).values
    res[f"V1real(mono@{a})_vs_sc8real(A@{b})"] = dict(tokens=int(X.shape[0]), exact=int((d == 0).sum()), maxabs=float(d.max()), ref_scale=float(X.abs().max()))
    print("PREFILL_CMP " + json.dumps(res)); raise SystemExit

tag = os.environ["TAG"]
AT = os.environ.get("AT_CONFIGS")
t = AutoTuner.get()
if AT:
    print("load_configs", t.load_configs(AT))
_orig = t.search_cache; seen = {}
def _sc(custom_op, runners, input_shapes, tuning_config, inputs=None):
    r = _orig(custom_op, runners, input_shapes, tuning_config, inputs=inputs)
    k = (custom_op, input_shapes[0] if input_shapes else None)
    if k not in seen: seen[k] = 1; print("TACTIC", tag, custom_op.split("::")[-1], input_shapes[0], "hit", r[0], "tactic", r[2])
    return r
t.search_cache = _sc

W = torch.load("/w/capture/L3_weights.pt", map_location=dev)
bias = W["bias"].float().to(dev); act = W["activation_int"]
caps = sorted(f for f in glob.glob("/w/capture/L3_*.pt") if "weights" not in f)
xq = torch.cat([torch.load(f, map_location=dev)["xq"] for f in caps]); xs = torch.cat([torch.load(f, map_location=dev)["xs"] for f in caps])
lg = torch.cat([torch.load(f, map_location=dev)["logits"].float() for f in caps])
n0 = xq.shape[0]; reps = (MT + n0 - 1) // n0
xq, xs, lg = xq.repeat(reps, 1)[:MT].contiguous(), xs.repeat(reps, 1)[:MT].contiguous(), lg.repeat(reps, 1)[:MT].contiguous()
print("tiled", n0, "real tokens ->", MT)

def mono(x, xs_, logits):
    out = flashinfer.fused_moe.trtllm_fp4_block_scale_moe(
        routing_logits=logits, routing_bias=bias, hidden_states=x, hidden_states_scale=xs_,
        gemm1_weights=W["w13"], gemm1_weights_scale=W["w13s"].view(torch.float8_e4m3fn), gemm1_bias=None,
        gemm1_alpha=None, gemm1_beta=None, gemm1_clamp_limit=None,
        gemm2_weights=W["w2"], gemm2_weights_scale=W["w2s"].view(torch.float8_e4m3fn), gemm2_bias=None,
        output1_scale_scalar=W["g1c"], output1_scale_gate_scalar=W["g1a"], output2_scale_scalar=W["g2a"],
        num_experts=E, top_k=K, n_group=1, topk_group=1, intermediate_size=INTER, local_expert_offset=0,
        local_num_experts=E, routed_scaling_factor=2.5, routing_method_type=int(RoutingMethodType.DeepSeekV3),
        do_finalize=True, activation_type=act, per_token_scale=None, tune_max_num_tokens=8192)
    return out[0] if isinstance(out, (list, tuple)) else out

def routed(x, xs_, topk_arg):
    o = torch.empty(x.shape[0], HID, device=dev, dtype=torch.bfloat16)
    flashinfer.fused_moe.trtllm_fp4_block_scale_routed_moe(
        topk_ids=topk_arg, routing_bias=None, hidden_states=x, hidden_states_scale=xs_,
        gemm1_weights=W["w13"], gemm1_weights_scale=W["w13s"].view(torch.float8_e4m3fn), gemm1_bias=None,
        gemm1_alpha=None, gemm1_beta=None, gemm1_clamp_limit=None,
        gemm2_weights=W["w2"], gemm2_weights_scale=W["w2s"].view(torch.float8_e4m3fn), gemm2_bias=None,
        output1_scale_scalar=W["g1c"], output1_scale_gate_scalar=W["g1a"], output2_scale_scalar=W["g2a"],
        num_experts=E, top_k=K, n_group=0, topk_group=0, intermediate_size=INTER, local_expert_offset=0,
        local_num_experts=E, routed_scaling_factor=None, routing_method_type=1, do_finalize=True,
        activation_type=act, per_token_scale=None, output=o, tune_max_num_tokens=8192)
    return o

w, i = grouped_topk(xq, lg, K, True, 1, 1, "sigmoid", 2.5, bias)
A = routed(xq, xs, trtllm_moe_pack_topk_ids_weights(i.to(torch.int32).contiguous(), w.float().contiguous()))
M_ = mono(xq, xs, lg)
# determinism check at this M
M2 = mono(xq, xs, lg); A2 = routed(xq, xs, trtllm_moe_pack_topk_ids_weights(i.to(torch.int32).contiguous(), w.float().contiguous()))
print("SELF", tag, "mono", float((M_.float() - M2.float()).abs().max()), "A", float((A.float() - A2.float()).abs().max()))
# batch-composition check: token 0 alone vs token 0 inside the big batch
m1 = mono(xq[:1].contiguous(), xs[:1].contiguous(), lg[:1].contiguous())
print("BATCHDEP", tag, "mono token0 M=1 vs M=%d maxabs" % MT, float((m1.float()[0] - M_.float()[0]).abs().max()))
torch.save(M_.float().cpu(), f"/w/capture/pf_mono_{tag}_M{MT}.pt"); torch.save(A.float().cpu(), f"/w/capture/pf_A_{tag}_M{MT}.pt")
print("PREFILL_RUN", json.dumps(dict(tag=tag, M=MT, mono_vs_A_exact=int(((M_.float() - A.float()).abs().max(dim=1).values == 0).sum()), maxabs=float((M_.float() - A.float()).abs().max()))))
