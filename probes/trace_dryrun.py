#!/usr/bin/env python3
"""trace_dryrun.py — exercise layer_trace hooks under CUDA graph capture/replay on a synthetic 3-layer model."""
import os, sys, torch, torch.nn as nn, importlib.util
os.environ["LAYER_TRACE_MINM"] = "1"
sp = importlib.util.spec_from_file_location("layer_trace", "/w/layer_trace.py"); lt = importlib.util.module_from_spec(sp); sp.loader.exec_module(lt)

class MLP(nn.Module):
    def __init__(s, d): super().__init__(); s.shared_experts = nn.Linear(d, d); s.experts = nn.Linear(d, d); s.gate = nn.Linear(d, 4)
    def forward(s, x): return s.shared_experts(x) + s.experts(x) + s.gate(x).sum(-1, keepdim=True)
class Layer(nn.Module):
    def __init__(s, d): super().__init__(); s.input_layernorm = nn.LayerNorm(d); s.self_attn = nn.Linear(d, d); s.post_attention_layernorm = nn.LayerNorm(d); s.mlp = MLP(d)
    def forward(s, x): return s.mlp(s.post_attention_layernorm(s.self_attn(s.input_layernorm(x)))), x
class Model(nn.Module):
    def __init__(s, d, n): super().__init__(); s.model = nn.Module(); s.model.layers = nn.ModuleList([Layer(d) for _ in range(n)])
    def forward(s, x):
        for l in s.model.layers: x, _ = l(x)
        return x
m = Model(64, 3).cuda().bfloat16()
lt._install(m)
x = torch.randn(16, 64, device="cuda", dtype=torch.bfloat16)
y0 = m(x)                       # eager: allocates the records
torch.cuda.synchronize()
s = torch.cuda.Stream()
with torch.cuda.stream(s):
    for _ in range(2): m(x)     # warmup on side stream
torch.cuda.current_stream().wait_stream(s)
g = torch.cuda.CUDAGraph()
xs = x.clone()
with torch.cuda.graph(g):
    ys = m(xs)
torch.cuda.synchronize()
xs.copy_(x * 2); g.replay(); torch.cuda.synchronize()
rec = lt._rec
name = "L01.mlp.experts"
st = rec[name]["stats"].tolist(); print("after replay x*2:", name, "sum=%.3f n=%d" % (st[0], int(rec[name]['n'])))
xs.copy_(x); g.replay(); torch.cuda.synchronize()
st2 = rec[name]["stats"].tolist(); print("after replay x:", name, "sum=%.3f n=%d" % (st2[0], int(rec[name]['n'])))
print("graph-safe:", st != st2 and int(rec[name]['n']) >= 5, "| modules:", len(lt._order), "| full:", sorted(lt._full)[:3])
os.makedirs("/wcap", exist_ok=True); open("/wcap/DUMP_TRACE", "w").close()
import time; time.sleep(7)
d = torch.load("/wcap/layer_trace.pt"); print("dump:", len(d["rec"]), "modules; order[:4]", d["order"][:4])
