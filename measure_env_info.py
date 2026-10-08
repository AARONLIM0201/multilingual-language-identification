import sys, os, re, json, collections, importlib

def try_version(modname):
    try:
        m = importlib.import_module(modname)
        return getattr(m, "__version__", "<no __version__>")
    except Exception:
        return None


# Python version
print(sys.version.replace("\n", " "))

# torch
torch_ver = try_version("torch")
print(torch_ver if torch_ver is not None else "torch: not installed")

# transformers
transf_ver = try_version("transformers")
print(transf_ver if transf_ver is not None else "transformers: not installed")

# easyocr (optional)
easyocr_ver = try_version("easyocr")
print(easyocr_ver if easyocr_ver is not None else "easyocr: not installed")

# sklearn
sklearn_ver = try_version("sklearn")
print(sklearn_ver if sklearn_ver is not None else "sklearn: not installed")


# GPU name and VRAM (if CUDA available)
try:
    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        total_mem = props.total_memory
        print(f"{name} | {total_mem}")
    else:
        print("CUDA not available")
except Exception:
    print("torch CUDA check failed")

# Total system RAM
def total_ram_bytes():
    try:
        import psutil
        return psutil.virtual_memory().total
    except Exception:
        try:
            import ctypes
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ('dwLength', ctypes.c_ulong),
                    ('dwMemoryLoad', ctypes.c_ulong),
                    ('ullTotalPhys', ctypes.c_ulonglong),
                    ('ullAvailPhys', ctypes.c_ulonglong),
                    ('ullTotalPageFile', ctypes.c_ulonglong),
                    ('ullAvailPageFile', ctypes.c_ulonglong),
                    ('ullTotalVirtual', ctypes.c_ulonglong),
                    ('ullAvailVirtual', ctypes.c_ulonglong),
                    ('sullAvailExtendedVirtual', ctypes.c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return stat.ullTotalPhys
        except Exception:
            return None

ram = total_ram_bytes()
print(str(ram) if ram is not None else "Unknown")