#!/usr/bin/env python3
"""router3.py — same fp32 gate logits, three routers: vLLM grouped_topk (Modular's), TRT fused_topk_deepseek
(standalone), Monolithic kernel's own routing (routing_replay_out). Report expert-SET disagreements per token
and the margin at the 8th/9th expert where they happen. Also compare the ids the LIVE server chose (from the ring).
"""
import torch, json
import flashinfer
from flashinfer.fused_moe import RoutingMethodType
from flashinfer.fused_moe.fused_routing_dsv3 import get_dsv3_fused_routing_module
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import grouped_topk

dev = "cuda"; E, K, HID, INTER = 256, 8, 6144, 2048
R = torch.load("/w/capture/L3_logit_ring.pt")
n = min(R["n"], R["ring"]); logits = R["logits"][:n].to(dev); live_ids = R["ids"][:n].to(dev)
W = torch.load("/w/capture/L3_weights.pt", map_location=dev); bias = W["bias"].float().to(dev); act = W["activation_int"]
print("tokens:", n)

def route_vllm(lg):
    x = torch.zeros(lg.shape[0], HID, dtype=torch.bfloat16, device=dev)
    w, i = grouped_topk(x, lg, K, True, 1, 1, "sigmoid", 1.0, bias)   # scale 1.0: GLM applies 2.5 on output
    return i.to(torch.int32), w.float()
_d = get_dsv3_fused_routing_module()
def route_trt(lg):
    M = lg.shape[0]; tv = torch.empty(M, K, dtype=torch.float32, device=dev); ti = torch.empty(M, K, dtype=torch.int32, device=dev)
    _d.NoAuxTc(lg.contiguous(), bias, 1, 1, K, 1.0, tv, ti, False, None); return ti, tv
def route_mono(lg):
    M = lg.shape[0]; rr = torch.full((M, K), -1, dtype=torch.int16, device=dev)
    x = torch.zeros(M, HID // 2, dtype=torch.uint8, device=dev); xs = torch.zeros(M, HID // 16, dtype=torch.float8_e4m3fn, device=dev)
    flashinfer.fused_moe.trtllm_fp4_block_scale_moe(routing_logits=lg.contiguous(), routing_bias=bias, hidden_states=x, hidden_states_scale=xs,
        gemm1_weights=W["w13"], gemm1_weights_scale=W["w13s"].view(torch.float8_e4m3fn), gemm1_bias=None, gemm1_alpha=None, gemm1_beta=None, gemm1_clamp_limit=None,
        gemm2_weights=W["w2"], gemm2_weights_scale=W["w2s"].view(torch.float8_e4m3fn), gemm2_bias=None,
        output1_scale_scalar=W["g1c"], output1_scale_gate_scalar=W["g1a"], output2_scale_scalar=W["g2a"],
        num_experts=E, top_k=K, n_group=1, topk_group=1, intermediate_size=INTER, local_expert_offset=0, local_num_experts=E,
        routed_scaling_factor=1.0, routing_method_type=int(RoutingMethodType.DeepSeekV3), do_finalize=True, activation_type=act,
        per_token_scale=None, tune_max_num_tokens=8, routing_replay_out=rr)
    return rr.to(torch.int32)

out = {}
B = 512
sets = {"vllm": [], "trt": [], "mono": []}
for s in range(0, n, B):
    lg = logits[s:s+B]
    iv, _ = route_vllm(lg); it, _ = route_trt(lg); im = route_mono(lg)
    sets["vllm"].append(torch.sort(iv, 1)[0]); sets["trt"].append(torch.sort(it, 1)[0]); sets["mono"].append(torch.sort(im, 1)[0])
for k in sets: sets[k] = torch.cat(sets[k])
live = torch.sort(live_ids, 1)[0]
def mism(a, b): return int((a != b).any(1).sum())
out["tokens"] = n
out["vllm_vs_mono"] = mism(sets["vllm"], sets["mono"]); out["trt_vs_mono"] = mism(sets["trt"], sets["mono"]); out["vllm_vs_trt"] = mism(sets["vllm"], sets["trt"])
out["live_vs_vllm_offline"] = mism(live, sets["vllm"]); out["live_vs_mono"] = mism(live, sets["mono"])
# margin analysis on the disagreeing tokens: sigmoid(logit)+bias scores, gap between 8th and 9th
sc = torch.sigmoid(logits) + bias[None, :]
top9 = torch.topk(sc, 9, dim=1).values
gap = (top9[:, 7] - top9[:, 8])
bad = (sets["vllm"] != sets["mono"]).any(1)
out["gap_8th_9th_median_all"] = float(gap.median()); out["gap_8th_9th_max_on_disagreeing"] = float(gap[bad].max()) if bad.any() else None
out["n_gap_below_1e-6"] = int((gap < 1e-6).sum()); out["n_exact_ties"] = int((gap == 0).sum())
if bad.any():
    i = int(bad.nonzero()[0]); a = sets["vllm"][i].tolist(); b = sets["mono"][i].tolist()
    out["example"] = dict(token=i, vllm=a, mono=b, only_vllm=sorted(set(a)-set(b)), only_mono=sorted(set(b)-set(a)), gap=float(gap[i]),
                          scores_around=[float(v) for v in torch.topk(sc[i], 10).values[6:10]])
print("ROUTER3 " + json.dumps(out))
