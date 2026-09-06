#!/usr/bin/env python3
"""ablate_layer.py <L> — offline two-class comparison on layer L with the REAL teacher-forced MLP input captured by layer_trace.
Builds layer-L kernel-format weights from the checkpoint through vLLM's stock pipeline (bit-exact vs the hook at layer 3),
quantizes the captured input exactly as vLLM's prepare() does, then runs:
   MONO  = trtllm_fp4_block_scale_moe (in-kernel DeepSeekV3 routing; routing_replay_out captured)
   ROUTED= NoAuxTc(1,1) -> pack bf16 -> trtllm_fp4_block_scale_routed_moe   (what sc9 runs)
and reports per-token: output maxabs, differing rows, expert-set mismatch, and the 8th/9th score gap on differing rows."""
import sys, json, re, torch
from safetensors import safe_open
import flashinfer
from flashinfer.fused_moe import RoutingMethodType
from flashinfer.fused_moe.fused_routing_dsv3 import get_dsv3_fused_routing_module
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import convert_to_nvfp4_moe_kernel_format, NvFp4MoeBackend, make_nvfp4_moe_quant_config
from vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe import TrtLlmNvFp4ExpertsMonolithic
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig, FusedMoEParallelConfig, RoutingMethodType as VRM, MoEActivation
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input, trtllm_moe_pack_topk_ids_weights as pack
from vllm._custom_ops import scaled_fp4_quant

L = int(sys.argv[1]); dev = "cuda"; E, K, HID, INTER = 256, 8, 6144, 2048
import os
TRACE = os.environ.get("TRACE_PT", "/w/trace/tf-sc9b/layer_trace.pt"); TRACE_B = os.environ.get("TRACE_PT_B")
T = torch.load(TRACE)
x_in = T["full"][f"L{L:02d}.mlp.gate.in"].to(dev)          # [M, 6144] bf16, MLP input
logits_live = T["full"][f"L{L:02d}.mlp.gate"].to(dev).float()
live_out = T["full"][f"L{L:02d}.mlp.experts"].to(dev)     # live routed-experts output from this trace (A)
live_out_b = torch.load(TRACE_B)["full"][f"L{L:02d}.mlp.experts"].to(dev) if TRACE_B else None
M = x_in.shape[0]; print(f"layer {L}: M={M} x_in {tuple(x_in.shape)} {x_in.dtype}; live gate logits {tuple(logits_live.shape)}")

# --- weights from checkpoint
idx = json.load(open("/model/model.safetensors.index.json"))["weight_map"]
pref = f"model.layers.{L}.mlp.experts."
files = sorted({idx[k] for k in idx if k.startswith(pref)})
raw = {}
for f in files:
    with safe_open(f"/model/{f}", "pt", device="cpu") as fh:
        for k in fh.keys():
            if k.startswith(pref): raw[k] = fh.get_tensor(k)
def stack(n): return torch.stack([raw[f"{pref}{e}.{n}"] for e in range(E)])
w1, w3, w2 = stack("gate_proj.weight"), stack("up_proj.weight"), stack("down_proj.weight")
w1s, w3s, w2s = stack("gate_proj.weight_scale"), stack("up_proj.weight_scale"), stack("down_proj.weight_scale")
w1s2, w3s2, w2s2 = stack("gate_proj.weight_scale_2").reshape(E), stack("up_proj.weight_scale_2").reshape(E), stack("down_proj.weight_scale_2").reshape(E)
a1, a3, a2 = stack("gate_proj.input_scale").reshape(E), stack("up_proj.input_scale").reshape(E), stack("down_proj.input_scale").reshape(E)
gate_w = None
for k in idx:
    if k == f"model.layers.{L}.mlp.gate.weight":
        with safe_open(f"/model/{idx[k]}", "pt", device="cpu") as fh: gate_w = fh.get_tensor(k)
    if k == f"model.layers.{L}.mlp.gate.e_score_correction_bias":
        with safe_open(f"/model/{idx[k]}", "pt", device="cpu") as fh: bias = fh.get_tensor(k)
bias = bias.float().to(dev)
# gate logits recomputed in fp32 like the model (x.float() @ W.T)
logits = torch.nn.functional.linear(x_in.float(), gate_w.float().to(dev))
print("gate logits recomputed vs live: maxabs", float((logits - logits_live).abs().max()))

_pc = FusedMoEParallelConfig(tp_size=1, pcp_size=1, dp_size=1, ep_size=1, tp_rank=0, pcp_rank=0, dp_rank=0, ep_rank=0, sp_size=1, use_ep=False, all2all_backend="naive", enable_eplb=False)
_mc = FusedMoEConfig(num_experts=E, experts_per_token=K, hidden_dim=HID, intermediate_size=INTER, num_local_experts=E, num_logical_experts=E,
                     activation=MoEActivation.SILU, device=torch.device(dev), routing_method=VRM.DeepSeekV3, moe_parallel_config=_pc,
                     in_dtype=torch.bfloat16, router_logits_dtype=torch.float32)
class Lyr(torch.nn.Module):
    def __init__(s): super().__init__(); s.activation = MoEActivation.SILU; s.moe_config = _mc
layer = Lyr()
P = lambda t: torch.nn.Parameter(t.to(dev), requires_grad=False)
layer.w13_weight = P(torch.cat([w1, w3], 1)); layer.w13_weight_scale = P(torch.cat([w1s, w3s], 1)); layer.w13_weight_scale_2 = P(torch.stack([w1s2, w3s2], 1))
layer.w13_input_scale = P(torch.maximum(a1, a3)); layer.w2_weight = P(w2); layer.w2_weight_scale = P(w2s); layer.w2_weight_scale_2 = P(w2s2); layer.w2_input_scale = P(a2)
out = convert_to_nvfp4_moe_kernel_format(nvfp4_backend=NvFp4MoeBackend.FLASHINFER_TRTLLM, layer=layer, w13=layer.w13_weight, w13_scale=layer.w13_weight_scale,
        w13_scale_2=layer.w13_weight_scale_2[:, 0].contiguous(), a13_scale=layer.w13_input_scale, w2=layer.w2_weight, w2_scale=layer.w2_weight_scale,
        w2_scale_2=layer.w2_weight_scale_2, a2_scale=layer.w2_input_scale, is_act_and_mul=True)
for n, t in zip(["w13_weight","w13_weight_scale","w13_weight_scale_2","w13_input_scale","w2_weight","w2_weight_scale","w2_weight_scale_2","w2_input_scale"], out):
    setattr(layer, n, torch.nn.Parameter(t, requires_grad=False))
qc = make_nvfp4_moe_quant_config(backend=NvFp4MoeBackend.FLASHINFER_TRTLLM, w13_scale=layer.w13_weight_scale, w2_scale=layer.w2_weight_scale,
        w13_scale_2=layer.w13_weight_scale_2, w2_scale_2=layer.w2_weight_scale_2, a13_scale=layer.w13_input_scale, a2_scale=layer.w2_input_scale,
        swiglu_limit=None, swiglu_alpha=None, swiglu_beta=None, layer=layer)
ex = TrtLlmNvFp4ExpertsMonolithic(moe_config=_mc, quant_config=qc); ex.process_weights_after_loading(layer)
W13, W13S, W2, W2S = layer.w13_weight.data, layer.w13_weight_scale.data.view(torch.float8_e4m3fn), layer.w2_weight.data, layer.w2_weight_scale.data.view(torch.float8_e4m3fn)
g1c, g1a, g2a = layer.g1_scale_c.data, qc.g1_alphas, qc.g2_alphas
print("weights built; g1c/g1a/g2a", tuple(g1c.shape), "a1_gscale", tuple(qc.a1_gscale.shape), float(qc.a1_gscale.flatten()[0]))

# --- activation quant exactly like prepare(): scaled_fp4_quant(x, a1_gscale, swizzled)
xq, xs = moe_kernel_quantize_input(x_in, qc.a1_gscale, quant_dtype="nvfp4", per_act_token_quant=False, block_shape=None, is_scale_swizzled=qc.is_scale_swizzled)
xs_k = xs.view(torch.float8_e4m3fn).reshape(M, -1)
act = 0  # SwiGlu — same int the hook records (_act_int); verify against W["activation_int"] from the layer-3 capture
try: act = int(torch.load("/w/capture/L3_weights.pt", map_location="cpu")["activation_int"])
except Exception: pass

rr = torch.full((M, K), -1, dtype=torch.int16, device=dev)
mono = flashinfer.fused_moe.trtllm_fp4_block_scale_moe(routing_logits=logits.contiguous(), routing_bias=bias, hidden_states=xq, hidden_states_scale=xs_k,
    gemm1_weights=W13, gemm1_weights_scale=W13S, gemm1_bias=None, gemm1_alpha=None, gemm1_beta=None, gemm1_clamp_limit=None,
    gemm2_weights=W2, gemm2_weights_scale=W2S, gemm2_bias=None, output1_scale_scalar=g1c, output1_scale_gate_scalar=g1a, output2_scale_scalar=g2a,
    num_experts=E, top_k=K, n_group=1, topk_group=1, intermediate_size=INTER, local_expert_offset=0, local_num_experts=E,
    routed_scaling_factor=1.0, routing_method_type=int(RoutingMethodType.DeepSeekV3), do_finalize=True, activation_type=act,
    per_token_scale=None, tune_max_num_tokens=max(8, M), routing_replay_out=rr)
mono = mono[0] if isinstance(mono, (list, tuple)) else mono
_d = get_dsv3_fused_routing_module()
tv = torch.empty(M, K, dtype=torch.float32, device=dev); ti = torch.empty(M, K, dtype=torch.int32, device=dev)
_d.NoAuxTc(logits.contiguous(), bias, 1, 1, K, 1.0, tv, ti, False, None)
o = torch.empty(M, HID, dtype=torch.bfloat16, device=dev)
flashinfer.fused_moe.trtllm_fp4_block_scale_routed_moe(topk_ids=pack(ti, tv), routing_bias=None, hidden_states=xq, hidden_states_scale=xs_k,
    gemm1_weights=W13, gemm1_weights_scale=W13S, gemm1_bias=None, gemm1_alpha=None, gemm1_beta=None, gemm1_clamp_limit=None,
    gemm2_weights=W2, gemm2_weights_scale=W2S, gemm2_bias=None, output1_scale_scalar=g1c, output1_scale_gate_scalar=g1a, output2_scale_scalar=g2a,
    num_experts=E, top_k=K, n_group=0, topk_group=0, intermediate_size=INTER, local_expert_offset=0, local_num_experts=E,
    routed_scaling_factor=None, routing_method_type=1, do_finalize=True, activation_type=act, per_token_scale=None, output=o, tune_max_num_tokens=max(8, M))
d = (mono.float() - o.float()).abs(); rows = (d > 0).any(1).nonzero().flatten().tolist()
ids_m = torch.sort(rr.to(torch.int32), 1)[0]; ids_r = torch.sort(ti, 1)[0]; set_mism = (ids_m != ids_r).any(1).nonzero().flatten().tolist()
sc = torch.sigmoid(logits) + bias[None]; top9 = torch.topk(sc, 9, 1).values; gap = (top9[:, 7] - top9[:, 8])
res = {"layer": L, "M": M, "out_maxabs": float(d.max()), "rows_differ": rows[:20], "n_rows_differ": len(rows), "expert_set_mismatch_rows": set_mism[:20],
       "gap_on_differing_rows": [float(gap[r]) for r in rows[:8]], "n_exact_ties": int((gap == 0).sum()), "gap_min": float(gap.min())}
if rows:
    r = rows[0]; res["example"] = {"row": r, "mono": ids_m[r].tolist(), "routed": ids_r[r].tolist(), "row_maxabs": float(d[r].max()),
                                   "mono_weights_bf16": [float(v) for v in torch.sort(tv[r])[0].bfloat16().float()[:8]]}
# --- compare both offline kernels against the LIVE outputs captured in the traces
def cmp(a, b): dd = (a.float() - b.float()).abs(); return {"neq": int((dd > 0).sum()), "maxabs": float(dd.max())}
res["live_A_vs_mono_offline"] = cmp(live_out, mono); res["live_A_vs_routed_offline"] = cmp(live_out, o)
if live_out_b is not None:
    res["live_B_vs_mono_offline"] = cmp(live_out_b, mono); res["live_B_vs_routed_offline"] = cmp(live_out_b, o); res["live_A_vs_live_B"] = cmp(live_out, live_out_b)
print("ABLATE_LAYER " + json.dumps(res))
