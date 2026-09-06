"""layer_trace.py — per-module output fingerprints for a teacher-forced comparison between two vLLM servers.

Activated by LAYER_TRACE=1 (loaded from sitecustomize on both V1 and slot-cache builds). Registers forward hooks on every
Glm4MoeDecoderLayer and its sub-modules (input_layernorm, self_attn, post_attention_layernorm, mlp, mlp.shared_experts,
mlp.experts). Each hook records, on device and graph-safe (fixed shapes, no allocation, no sync):
  sum(fp64), sum|x|(fp64), sum x^2 (fp64), and the first 64 elements' bf16 bits of the (first) output tensor,
plus M (rows). Only calls with M >= LAYER_TRACE_MINM (default 8) are recorded, so the teacher-forced prefill request is
the one that lands. Dump on sentinel file /wcap/DUMP_TRACE -> /wcap/layer_trace.pt (+ full tensors for layers <= LAYER_TRACE_FULL).
"""
import os, sys, threading, time, torch

_MINM = int(os.environ.get("LAYER_TRACE_MINM", "8"))
_FULL = int(os.environ.get("LAYER_TRACE_FULL", "5"))
_LOG = lambda m: sys.stderr.write(f"LAYER_TRACE {m}\n")
_rec = {}          # name -> dict(stats=tensor[5] f64, head=tensor[64] int16, m=tensor[] int64, n=tensor[] int64)
_full = {}         # name -> tensor (clone of output for small layers, taken outside capture)
_order = []
_lock = threading.Lock()

def _first_tensor(out):
    if isinstance(out, torch.Tensor): return out
    if isinstance(out, (tuple, list)):
        for o in out:
            if isinstance(o, torch.Tensor): return o
    return None

def _make_hook(name, layer_idx):
    def hook(mod, inp, out):
        t = _first_tensor(out)
        if t is None or t.dim() < 2: return
        M = t.shape[0]
        if M < _MINM: return
        r = _rec.get(name)
        if r is None:
            if torch.cuda.is_current_stream_capturing(): return
            r = dict(stats=torch.zeros(3, dtype=torch.float64, device=t.device), head=torch.zeros(64, dtype=torch.int16, device=t.device),
                     m=torch.zeros((), dtype=torch.int64, device=t.device), n=torch.zeros((), dtype=torch.int64, device=t.device))
            with _lock:
                _rec[name] = r; _order.append(name)
        x = t.detach()
        xf = x.to(torch.float64)
        r["stats"][0] = xf.sum(); r["stats"][1] = xf.abs().sum(); r["stats"][2] = (xf * xf).sum()
        flat = x.reshape(-1)
        k = min(64, flat.numel())
        r["head"][:k] = flat[:k].to(torch.bfloat16).view(torch.int16)
        r["m"].fill_(M); r["n"] += 1
        if layer_idx is not None and layer_idx <= _FULL and not torch.cuda.is_current_stream_capturing():
            _full[name] = x.clone()
    return hook

def _install(model):
    import torch.nn as nn
    layers = None
    for mname, m in model.named_modules():
        if mname.endswith("model.layers") or mname == "layers":
            layers = m; break
    if layers is None:
        for mname, m in model.named_modules():
            if isinstance(m, nn.ModuleList) and len(m) >= 70: layers = m; break
    if layers is None:
        _LOG("no decoder layer list found"); return
    n = 0
    for i, layer in enumerate(layers):
        subs = [("", layer), ("input_layernorm", getattr(layer, "input_layernorm", None)), ("self_attn", getattr(layer, "self_attn", None)),
                ("post_attention_layernorm", getattr(layer, "post_attention_layernorm", None)), ("mlp", getattr(layer, "mlp", None))]
        mlp = getattr(layer, "mlp", None)
        if mlp is not None:
            subs += [("mlp.shared_experts", getattr(mlp, "shared_experts", None)), ("mlp.experts", getattr(mlp, "experts", None)),
                     ("mlp.gate", getattr(mlp, "gate", None))]
        for sname, sm in subs:
            if sm is None or not isinstance(sm, nn.Module): continue
            name = f"L{i:02d}" + (f".{sname}" if sname else "")
            sm.register_forward_hook(_make_hook(name, i)); n += 1
    _LOG(f"installed {n} hooks on {len(layers)} layers; minM={_MINM} full<=L{_FULL}")
    threading.Thread(target=_dumper, daemon=True).start()

def _dumper():
    while True:
        time.sleep(5)
        if os.path.exists("/wcap/DUMP_TRACE"):
            try:
                torch.cuda.synchronize()
                out = {"order": list(_order),
                       "rec": {k: {kk: vv.cpu() for kk, vv in v.items()} for k, v in _rec.items()},
                       "full": {k: v.cpu() for k, v in _full.items()}}
                torch.save(out, os.environ.get("LAYER_TRACE_OUT", "/wcap/layer_trace.pt"))
                _LOG(f"dumped {len(_rec)} modules, {len(_full)} full tensors")
            except Exception as e:
                _LOG(f"dump failed: {e!r}")
            try: os.remove("/wcap/DUMP_TRACE")
            except Exception: pass

def install():
    # Seam: BaseModelLoader.load_model returns the constructed model with weights loaded (gpu_model_runner.load_model:5435).
    from vllm.model_executor.model_loader.base_loader import BaseModelLoader
    orig = BaseModelLoader.load_model
    def load_model(self, *a, **k):
        model = orig(self, *a, **k)
        try: _install(model)
        except Exception as e: _LOG(f"install failed: {e!r}")
        return model
    BaseModelLoader.load_model = load_model
    _LOG("armed")
