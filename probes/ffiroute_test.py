#!/usr/bin/env python3
"""ffiroute_test.py — validate ffi_route.mono_route against Monolithic on all 3041 real layer-3 tokens:
 (a) replay ids == Monolithic replay ids; (b) bf16 weights == Monolithic expert_weights (do_finalize=False);
 (c) packed -> routed kernel output == Monolithic finalized output (bit-exact); (d) timing vs NoAuxTc."""
import sys, json, time, torch, flashinfer
sys.path.insert(0, "/w")
from flashinfer.fused_moe import RoutingMethodType
from flashinfer.fused_moe.fused_routing_dsv3 import get_dsv3_fused_routing_module
import ffi_route
from vllm.model_executor.layers.fused_moe.utils import trtllm_moe_pack_topk_ids_weights as vpack
dev = "cuda"; E, K, HID, INTER = 256, 8, 6144, 2048
W = torch.load("/w/capture/L3_weights.pt", map_location=dev)
W13, W13S, W2, W2S, g1c, g1a, g2a = W["w13"], W["w13s"], W["w2"], W["w2s"], W["g1c"], W["g1a"], W["g2a"]
bias32 = W["bias"].float().contiguous(); bias16 = W["bias"].to(torch.bfloat16).contiguous()
act = int(W["activation_int"])
c0 = torch.load("/w/capture/L3_0000.pt", map_location=dev); xq, xs = c0["xq"][:1].contiguous(), c0["xs"][:1].contiguous()
ring = torch.load("/w/capture/L3_logit_ring.pt", map_location=dev); logits_all = ring["logits"].to(dev).float()[: int(ring["n"])]; n = logits_all.shape[0]
print("bias dtype in capture:", W["bias"].dtype, "logit ring dtype:", ring["logits"].dtype, flush=True)
_d = get_dsv3_fused_routing_module()
common = dict(hidden_states=xq, hidden_states_scale=xs, gemm1_weights=W13, gemm1_weights_scale=W13S, gemm1_bias=None, gemm1_alpha=None, gemm1_beta=None,
              gemm1_clamp_limit=None, gemm2_weights=W2, gemm2_weights_scale=W2S, gemm2_bias=None, output1_scale_scalar=g1c, output1_scale_gate_scalar=g1a,
              output2_scale_scalar=g2a, num_experts=E, top_k=K, intermediate_size=INTER, local_expert_offset=0, local_num_experts=E, activation_type=act,
              per_token_scale=None, tune_max_num_tokens=8)
def mono(lg, bias, finalize=True):
    rr = torch.full((1, K), -1, dtype=torch.int16, device=dev)
    y = flashinfer.fused_moe.trtllm_fp4_block_scale_moe(routing_logits=lg.contiguous(), routing_bias=bias, n_group=1, topk_group=1, routed_scaling_factor=2.5,
        routing_method_type=int(RoutingMethodType.DeepSeekV3), do_finalize=finalize, routing_replay_out=rr, **common)
    return y, rr
def routed(packed):
    o = torch.empty(1, HID, dtype=torch.bfloat16, device=dev)
    flashinfer.fused_moe.trtllm_fp4_block_scale_routed_moe(topk_ids=packed, routing_bias=None, n_group=0, topk_group=0, routed_scaling_factor=None,
        routing_method_type=1, do_finalize=True, output=o, **common)
    return o
ws = ffi_route.RouteWorkspace(1, K, E, torch.device(dev))
res = {}
for lbl, bias in (("bias_fp32", bias32), ("bias_bf16", bias16)):
    bad_ids = bad_w = bad_out = 0; first = None
    for t in range(n):
        lg = logits_all[t:t+1]
        ym, rm = mono(lg, bias, True); ym = ym[0] if isinstance(ym, (list, tuple)) else ym
        yf, rf = mono(lg, bias, False); wf = yf[1][:, :K]
        packed, wts, rr = ffi_route.mono_route(lg, bias, ws, routed_scaling_factor=2.5)
        torch.cuda.synchronize()
        if not torch.equal(rr, rm): bad_ids += 1
        if not torch.equal(wts, wf): bad_w += 1
        yr = routed(vpack(rr.to(torch.int32).contiguous(), wts.float().contiguous()))
        if not torch.equal(yr, ym):
            bad_out += 1
            if first is None: first = {"t": t, "maxabs": float((yr.float()-ym.float()).abs().max()), "ffi_ids": rr.tolist(), "mono_ids": rm.tolist(), "ffi_w": wts.float().tolist(), "mono_w": wf.float().tolist()}
    res[lbl] = {"ids_mismatch": bad_ids, "weights_mismatch": bad_w, "output_mismatch": bad_out, "first_bad": first}
    print("FFIROUTE", lbl, json.dumps(res[lbl]), flush=True)
def timeit(f, iters=500):
    for _ in range(20): f()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(iters): f()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / iters * 1e6
lg = logits_all[:1].contiguous()
tv = torch.empty(1, K, dtype=torch.float32, device=dev); ti = torch.empty(1, K, dtype=torch.int32, device=dev)
res["us_noauxtc"] = round(timeit(lambda: _d.NoAuxTc(lg, bias32, 1, 1, K, 2.5, tv, ti, False, None)), 1)
res["us_ffi_route"] = round(timeit(lambda: ffi_route.mono_route(lg, bias32, ws)), 1)
# graph capture test: is mono_route capturable?
try:
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): ffi_route.mono_route(lg, bias32, ws)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): ffi_route.mono_route(lg, bias32, ws)
    ws.packed.zero_(); g.replay(); torch.cuda.synchronize()
    packed_g = ws.packed.clone(); ffi_route.mono_route(lg, bias32, ws); torch.cuda.synchronize()
    res["graph_capture"] = "ok" if torch.equal(packed_g, ws.packed) else "replay_mismatch"
    res["us_ffi_route_graph"] = round(timeit(lambda: g.replay()), 1)
except Exception as e:
    res["graph_capture"] = f"ERR {str(e)[:160]}"
print("FFIROUTE_ALL", json.dumps(res), flush=True)
