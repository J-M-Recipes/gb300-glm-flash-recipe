#!/usr/bin/env python3
"""Where does host memory go across N simulated layers of: UVA-offload -> device_loading_context repack -> re-pin?
Reports shmem RSS + total RSS after each layer with the exact_pin shadow installed (same as the sc7 launch)."""
import os, sys, json, gc, torch
sys.path.insert(0, "/w")
import exact_pin
exact_pin.install_uva_patch()
from vllm.model_executor.model_loader import utils as lu
from vllm.model_executor.offloader.uva import UVAOffloader
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
def st():
    s = open(f"/proc/{os.getpid()}/status").read()
    g = lambda k: int(s.split(k + ":")[1].split()[0]) // 1024
    return g("RssShmem"), g("RssAnon"), g("VmRSS")
E, HID, I = 256, 6144, 2048
off = UVAOffloader(cpu_offload_max_bytes=1 << 50, cpu_offload_params={"w13_weight", "w2_weight"})
rows = []
mods = []
for layer in range(int(os.environ.get("NL", "6"))):
    m = torch.nn.Module()
    m.w13_weight = torch.nn.Parameter(torch.randint(0, 255, (E, 2 * I, HID // 2), dtype=torch.uint8, device="cuda"), requires_grad=False)
    m.w2_weight = torch.nn.Parameter(torch.randint(0, 255, (E, HID, I // 2), dtype=torch.uint8, device="cuda"), requires_grad=False)
    off._maybe_offload_to_cpu(m)                       # step 1: offload (exact pin via shadow)
    a = st()
    with lu.device_loading_context(m, torch.device("cuda")):   # step 2: repack (replace with fresh device tensors)
        m.w13_weight = torch.nn.Parameter(m.w13_weight.data.clone(), requires_grad=False)
        m.w2_weight = torch.nn.Parameter(m.w2_weight.data.clone(), requires_grad=False)
    b = st()
    mods.append(m); gc.collect()
    c = st()
    rows.append(dict(layer=layer, after_offload_shmem_GiB=round(a[0] / 1024, 2), after_repin_shmem_GiB=round(b[0] / 1024, 2),
                     after_gc_shmem_GiB=round(c[0] / 1024, 2), anon_GiB=round(c[1] / 1024, 2), rss_GiB=round(c[2] / 1024, 2),
                     w13_is_uva=bool(getattr(m.w13_weight, "_vllm_is_uva_offloaded", False)), w13_pinned=m.w13_weight.data.is_cuda))
per_layer_bytes = (E * 2 * I * HID // 2 + E * HID * I // 2) / 2**30
print("HOSTLEAK_TEST per_layer_real_GiB=%.2f" % per_layer_bytes)
for r in rows: print("HOSTLEAK_TEST " + json.dumps(r))
print("HOSTLEAK_TEST shmem_per_layer_GiB=%.2f (real %.2f)" % ((rows[-1]["after_gc_shmem_GiB"] - rows[0]["after_gc_shmem_GiB"]) / max(1, len(rows) - 1), per_layer_bytes))
