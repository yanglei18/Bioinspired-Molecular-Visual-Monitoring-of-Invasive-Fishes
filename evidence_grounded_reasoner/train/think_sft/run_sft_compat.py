#!/usr/bin/env python3
"""
Compatibility shim for ms-swift SFT on torch < 2.6 environments.
Patches torch.distributed.fsdp.FSDPModule (missing in torch 2.5.x) before
delegating to the standard ms-swift sft entry point.
Usage: invoked by train.sh via
    python3 run_sft_compat.py <all normal swift sft args>
"""
import sys
import types

# Patch 1: FSDPModule missing in torch 2.5.x (needed by swift 4.x)
try:
    from torch.distributed.fsdp import FSDPModule  # noqa: F401
except ImportError:
    import torch.distributed.fsdp as _fsdp
    _fsdp.FSDPModule = type("FSDPModule", (), {})

# Patch 2: matplotlib backend (non-fatal import sometimes fails)
try:
    import matplotlib
    matplotlib.use("Agg")
except Exception:
    pass

# Delegate to swift sft entry point
import runpy
runpy.run_module("swift.cli.sft", run_name="__main__", alter_sys=True)
