p = "/tmp/glm53doc/exact_pin.py"
s = open(p).read()
old = s[s.index("_keep = []"):s.index("def install_uva_patch():")]
new = '''import ctypes, weakref

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

'''
s = s.replace(old, new)
# the old top-level _check is inside the replaced block; make sure only one remains
assert s.count("def _check(") == 1, s.count("def _check(")
open(p, "w").write(s); print("exact_pin: finalizer-owned allocations")
