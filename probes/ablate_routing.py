#!/usr/bin/env python3
"""ablate_routing.py — 2x2 routing ablation on REAL captured layer inputs, offline in the vLLM image, no serving.

Captures, from a running Modular-path model... no: we cannot easily hook the live server. Instead this script
loads NOTHING large: it needs per-layer (router_logits fp32 [M,256], hidden_states bf16 [M,6144], a1q_scale) captured
by the capture hook (SLOT_CACHE_CAPTURE=1 in slot_cache_hook.py, writes /w/capture/<layer>_<step>.pt for the first
N decode steps), plus that layer's REAL weights re-read from the checkpoint... which is 5.4 GB/layer of NVFP4 slabs
we can load for ONE layer directly from safetensors + run process_weights_after_loading offline.

Simplification that keeps this honest: for the ablation we use ONE real MoE layer's real weights (layer 3, the
first MoE layer) and real captured inputs, and compare the four routing variants AGAINST MONOLITHIC on those exact
tensors. Monolithic is run via trtllm_fp4_block_scale_moe with the same weights (it routes internally).

Variants:
  A  vLLM router (grouped_topk fp32) + packed bf16 weights        (= today's Modular)
  B  vLLM router + unpacked (ids, fp32 weights)
  C  fused_topk_deepseek (TRT-LLM routing) + packed bf16
  D  fused_topk_deepseek + unpacked fp32
  M  Monolithic (reference)
Outputs: per-variant max-abs diff vs M over all captured tokens, count of tokens with ANY routing-id mismatch vs M
(M's ids come from routing_replay_out), and count of bit-exact tokens.
"""
import os, sys, json, glob, torch
sys.path.insert(0, "/w")
import flashinfer
from flashinfer.fused_moe import fused_topk_deepseek, RoutingMethodType
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import grouped_topk
from vllm.model_executor.layers.fused_moe.utils import trtllm_moe_pack_topk_ids_weights

dev = "cuda"
_AT = os.environ.get("AT_CONFIGS")
if _AT:
    from flashinfer.autotuner import AutoTuner
    ok = AutoTuner.get().load_configs(_AT)
    t = AutoTuner.get(); cands = {n: len(getattr(t, n)) for n in dir(t) if not n.startswith("__") and isinstance(getattr(t, n, None), dict)}
    print("AUTOTUNE load_configs ->", ok, "dict attrs:", cands)
    # log which tactic search_cache resolves for the M=1/2 trtllm call by wrapping search_cache
    _orig = t.search_cache
    _seen = {}
    def _sc(custom_op, runners, input_shapes, tuning_config, inputs=None):
        r = _orig(custom_op, runners, input_shapes, tuning_config, inputs=inputs)
        k = (custom_op, input_shapes[0] if input_shapes else None)
        if k not in _seen: _seen[k] = (r[0], r[1], r[2]); print("TACTIC", custom_op.split("::")[-1], input_shapes[0], "hit", r[0], "runner", r[1], "tactic", r[2])
        return r
    t.search_cache = _sc
TAG = os.environ.get("TAG", "L3"); cap = sorted(f for f in glob.glob(f"/w/capture/{TAG}_*.pt") if "weights" not in f)
assert cap, "no captures in /w/capture (run a server with SLOT_CACHE_CAPTURE=1 first)"
W = torch.load("/w/capture/L3_weights.pt", map_location=dev)   # kernel-format w13,w2,w13s,w2s,g1c,g1a,g2a,bias, dims
E, K, HID, INTER = 256, 8, 6144, 2048
bias = W["bias"].float().to(dev)
act = W["activation_int"]

def mono(x, xs, logits):
    out = flashinfer.fused_moe.trtllm_fp4_block_scale_moe(
        routing_logits=logits, routing_bias=bias, hidden_states=x, hidden_states_scale=xs,
        gemm1_weights=W["w13"], gemm1_weights_scale=W["w13s"].view(torch.float8_e4m3fn), gemm1_bias=None,
        gemm1_alpha=None, gemm1_beta=None, gemm1_clamp_limit=None,
        gemm2_weights=W["w2"], gemm2_weights_scale=W["w2s"].view(torch.float8_e4m3fn), gemm2_bias=None,
        output1_scale_scalar=W["g1c"], output1_scale_gate_scalar=W["g1a"], output2_scale_scalar=W["g2a"],
        num_experts=E, top_k=K, n_group=1, topk_group=1, intermediate_size=INTER, local_expert_offset=0,
        local_num_experts=E, routed_scaling_factor=2.5, routing_method_type=int(RoutingMethodType.DeepSeekV3),
        do_finalize=True, activation_type=act, per_token_scale=None, tune_max_num_tokens=int(os.environ.get("TUNE_MAX", "8")))
    return out[0] if isinstance(out, (list, tuple)) else out

def routed(x, xs, topk_arg):
    o = torch.empty(x.shape[0], HID, device=dev, dtype=torch.bfloat16)
    flashinfer.fused_moe.trtllm_fp4_block_scale_routed_moe(
        topk_ids=topk_arg, routing_bias=None, hidden_states=x, hidden_states_scale=xs,
        gemm1_weights=W["w13"], gemm1_weights_scale=W["w13s"].view(torch.float8_e4m3fn), gemm1_bias=None,
        gemm1_alpha=None, gemm1_beta=None, gemm1_clamp_limit=None,
        gemm2_weights=W["w2"], gemm2_weights_scale=W["w2s"].view(torch.float8_e4m3fn), gemm2_bias=None,
        output1_scale_scalar=W["g1c"], output1_scale_gate_scalar=W["g1a"], output2_scale_scalar=W["g2a"],
        num_experts=E, top_k=K, n_group=0, topk_group=0, intermediate_size=INTER, local_expert_offset=0,
        local_num_experts=E, routed_scaling_factor=None, routing_method_type=1, do_finalize=True,
        activation_type=act, per_token_scale=None, output=o, tune_max_num_tokens=int(os.environ.get("TUNE_MAX", "8")))
    return o

def route_vllm(logits, x):
    w, i = grouped_topk(x, logits, K, True, 1, 1, "sigmoid", 2.5, bias)
    return i.to(torch.int32).contiguous(), w.float().contiguous()

from flashinfer.fused_moe.fused_routing_dsv3 import get_dsv3_fused_routing_module
_dsv3 = get_dsv3_fused_routing_module()
def route_trt(logits, n_group=1, topk_group=1):
    # Bypass the Python validator (self-contradictory for n_group=1/top-8): call the compiled NoAuxTc op directly.
    M = logits.shape[0]
    tv = torch.empty(M, K, dtype=torch.float32, device=dev); ti = torch.empty(M, K, dtype=torch.int32, device=dev)
    _dsv3.NoAuxTc(logits, bias, n_group, topk_group, K, 2.5, tv, ti, False, None)
    return ti.contiguous(), tv.contiguous()

# Also read Monolithic's OWN routing decisions via routing_replay_out, if the kernel supports it in this build.
def mono_ids(x, xs, logits):
    M = logits.shape[0]
    rr = torch.full((M, K), -1, dtype=torch.int16, device=dev)
    try:
        out = flashinfer.fused_moe.trtllm_fp4_block_scale_moe(
            routing_logits=logits, routing_bias=bias, hidden_states=x, hidden_states_scale=xs,
            gemm1_weights=W["w13"], gemm1_weights_scale=W["w13s"].view(torch.float8_e4m3fn), gemm1_bias=None,
            gemm1_alpha=None, gemm1_beta=None, gemm1_clamp_limit=None,
            gemm2_weights=W["w2"], gemm2_weights_scale=W["w2s"].view(torch.float8_e4m3fn), gemm2_bias=None,
            output1_scale_scalar=W["g1c"], output1_scale_gate_scalar=W["g1a"], output2_scale_scalar=W["g2a"],
            num_experts=E, top_k=K, n_group=1, topk_group=1, intermediate_size=INTER, local_expert_offset=0,
            local_num_experts=E, routed_scaling_factor=2.5, routing_method_type=int(RoutingMethodType.DeepSeekV3),
            do_finalize=True, activation_type=act, per_token_scale=None, tune_max_num_tokens=int(os.environ.get("TUNE_MAX", "8")), routing_replay_out=rr)
        return rr
    except TypeError:
        return None

res = {k: dict(tokens=0, exact=0, maxabs=0.0, id_mismatch=0, weight_maxabs=0.0) for k in "ABCD"}
res["M_self"] = dict(tokens=0, exact=0, maxabs=0.0)
for f in cap:
    c = torch.load(f, map_location=dev)
    x, xs, logits = c["xq"], c["xs"], c["logits"].float()
    M = x.shape[0]
    ref = mono(x, xs, logits); ref2 = mono(x, xs, logits)
    d = (ref.float() - ref2.float()).abs().max().item(); res["M_self"]["tokens"] += M; res["M_self"]["exact"] += int(d == 0); res["M_self"]["maxabs"] = max(res["M_self"]["maxabs"], d)
    # Monolithic ids via routing replay are not exposed here; use TRT routing as proxy for M's ids (variant C/D) and
    # vLLM routing for A/B; report id mismatch between the two routers directly.
    iv, wv = route_vllm(logits, x); it, wt = route_trt(logits)
    ids_differ = int((torch.sort(iv, 1)[0] != torch.sort(it, 1)[0]).any(1).sum())
    mi = mono_ids(x, xs, logits)
    if mi is not None:
        res.setdefault("mono_ids", dict(tokens=0, vllm_set_mismatch=0, trt_set_mismatch=0))
        res["mono_ids"]["tokens"] += M
        res["mono_ids"]["vllm_set_mismatch"] += int((torch.sort(iv,1)[0] != torch.sort(mi,1)[0]).any(1).sum())
        res["mono_ids"]["trt_set_mismatch"] += int((torch.sort(it,1)[0] != torch.sort(mi,1)[0]).any(1).sum())
    # also: what did the LIVE vLLM router choose at capture time vs offline recompute (sanity)
    res.setdefault("live_vs_offline_vllm_router_mismatch", 0)
    res["live_vs_offline_vllm_router_mismatch"] += int((torch.sort(c["topk_ids"].to(torch.int32),1)[0] != torch.sort(iv,1)[0]).any(1).sum())
    wdiff = (torch.sort(wv, 1)[0] - torch.sort(wt, 1)[0]).abs().max().item()
    outs = {"A": routed(x, xs, trtllm_moe_pack_topk_ids_weights(iv, wv)), "B": routed(x, xs, (iv, wv)),
            "C": routed(x, xs, trtllm_moe_pack_topk_ids_weights(it, wt)), "D": routed(x, xs, (it, wt))}
    for k, o in outs.items():
        diff = (o.float() - ref.float()).abs().max(dim=1).values
        res[k]["tokens"] += M; res[k]["exact"] += int((diff == 0).sum()); res[k]["maxabs"] = max(res[k]["maxabs"], diff.max().item())
        res[k]["id_mismatch"] += ids_differ if k in "AB" else 0; res[k]["weight_maxabs"] = max(res[k]["weight_maxabs"], wdiff)
res["ref_scale"] = ref.float().abs().max().item()
tag = os.environ.get("MONO_DUMP"); 
if tag:
    outs_all = []
    for f in cap:
        c = torch.load(f, map_location=dev); outs_all.append(mono(c["xq"], c["xs"], c["logits"].float()).float().cpu())
    torch.save(outs_all, f"/w/capture/mono_{tag}.pt"); res["mono_dump"] = tag
cmp = os.environ.get("MONO_CMP")
if cmp:
    a, b = cmp.split(",")
    A = torch.load(f"/w/capture/mono_{a}.pt"); B = torch.load(f"/w/capture/mono_{b}.pt")
    tot = sum(x.shape[0] for x in A); ex = sum(int(((x - y).abs().max(dim=1).values == 0).sum()) for x, y in zip(A, B))
    mx = max((x - y).abs().max().item() for x, y in zip(A, B))
    res["mono_tactic_cmp"] = dict(a=a, b=b, tokens=tot, exact=ex, maxabs=mx)
res["legend"] = "A=vllm-router+packed(bf16) [today]  B=vllm-router+unpacked(fp32)  C=trt-routing+packed  D=trt-routing+unpacked; exact=tokens bit-identical to Monolithic; id_mismatch=tokens where vllm router picked a different expert SET than trt routing"
print("ABLATION " + json.dumps(res))
