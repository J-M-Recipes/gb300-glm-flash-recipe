#!/usr/bin/env python3
"""exact_pin.py — pin host memory with cudaHostAlloc (exact size) instead of torch.pin_memory() (rounds each block
up to a power of two: 3.22 GB -> 4 GiB, 33% waste on GLM expert slabs). Provides:
    exact_pinned_like(t) -> pinned CPU tensor with the same shape/dtype, exact allocation
    install_uva_patch()  -> patches vllm's UVA offloader + device_loading_context to use it
"""
import sys, torch
from cuda.bindings import runtime as rt

import ctypes, weakref

def _check(err):
    if err != rt.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cuda error {err}")

_live = {}     # ptr -> nbytes, for accounting only
def live_bytes(): return sum(_live.values())

def exact_pinned_like(src: torch.Tensor) -> torch.Tensor:
    """Pinned CPU tensor with src's shape/dtype, allocated with cudaHostAlloc at exact size.
    The host block is freed (cudaFreeHost) when the tensor's storage is garbage-collected — no global keepalive."""
    src = src.contiguous()
    nbytes = src.numel() * src.element_size()
    err, ptr = rt.cudaHostAlloc(max(nbytes, 1), rt.cudaHostAllocMapped | rt.cudaHostAllocPortable)
    _check(err)
    ptr = int(ptr)
    buf = (ctypes.c_uint8 * nbytes).from_address(ptr)
    t = torch.frombuffer(buf, dtype=torch.uint8, count=nbytes).view(src.dtype).reshape(src.shape)
    _live[ptr] = nbytes
    def _free(_p=ptr, _buf=buf):          # closure keeps the ctypes buffer alive exactly as long as the storage
        _live.pop(_p, None)
        rt.cudaFreeHost(_p)
    weakref.finalize(t.untyped_storage(), _free)
    t.copy_(src)
    return t

def install_uva_patch():
    import importlib.abc, importlib.util
    def _patch_uva(module):
        Off = next(c for c in vars(module).values() if isinstance(c, type) and hasattr(c, "_maybe_offload_to_cpu"))
        src = module  # noqa
        orig = Off._maybe_offload_to_cpu
        # monkeypatch torch.Tensor.pin_memory only within the offloader's call: simplest is to shadow the method
        def _maybe_offload_to_cpu(self, mod):
            real_pin = torch.Tensor.pin_memory
            def fake_pin(t, *a, **k):
                return exact_pinned_like(t)
            torch.Tensor.pin_memory = fake_pin
            try:
                return orig(self, mod)
            finally:
                torch.Tensor.pin_memory = real_pin
        Off._maybe_offload_to_cpu = _maybe_offload_to_cpu
        sys.stderr.write("EXACT_PIN uva offloader patched (cudaHostAlloc exact-size pinning)\n")
    def _patch_loader_utils(module):
        # device_loading_context's finally-block re-pins UVA params via cpu_data.pin_memory(). A contextmanager
        # wrapper cannot shadow across that boundary (the yield returns before finally runs), so patch the
        # method the finally block calls: shadow torch.Tensor.pin_memory for the WHOLE context, including the
        # body (the body only calls pin_memory in quant repack paths, where exact pinning is also fine).
        orig = module.device_loading_context
        import contextlib
        @contextlib.contextmanager
        def device_loading_context(mod, target_device):
            real_pin = torch.Tensor.pin_memory
            torch.Tensor.pin_memory = lambda t, *a, **k: exact_pinned_like(t)
            try:
                with orig(mod, target_device) as m:
                    yield m
            finally:
                torch.Tensor.pin_memory = real_pin
        module.device_loading_context = device_loading_context
        sys.stderr.write("EXACT_PIN device_loading_context patched (whole-context shadow)\n")
    targets = {"vllm.model_executor.offloader.uva": _patch_uva,
               "vllm.model_executor.model_loader.utils": _patch_loader_utils}
    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name not in targets: return None
            fn = targets.pop(name)
            if not targets:
                try: sys.meta_path.remove(self)
                except ValueError: pass
            spec = importlib.util.find_spec(name)
            if spec is None or spec.loader is None: return None
            loader = spec.loader; oe = loader.exec_module
            def exec_module(m, _o=oe, _f=fn): _o(m); _f(m)
            loader.exec_module = exec_module
            return spec
    sys.meta_path.insert(0, _Finder())

if __name__ == "__main__":
    import os
    def rss():
        return int(open(f"/proc/{os.getpid()}/status").read().split("RssShmem:")[1].split()[0]) // 1024
    x = torch.randint(0, 255, (256, 4096, 3072), dtype=torch.uint8)
    b = rss(); p = exact_pinned_like(x); d = rss() - b
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
    v = get_accelerator_view_from_cpu_tensor(p)
    ok = bool((v[3, :5, :5].cpu() == x[3, :5, :5]).all())
    print(f"EXACT_PIN_TEST {x.numel()/1e9:.2f} GB -> +{d/1024:.2f} GiB pinned; is_pinned={p.is_pinned()} uva_view_ok={ok} device={v.device}")
