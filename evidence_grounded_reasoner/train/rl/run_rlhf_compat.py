#!/usr/bin/env python3
"""
Minimal compatibility shim for running ms-swift GRPO RL on environments where
torch.distributed.fsdp does not yet expose FSDPModule (torch < 2.6).
Applies only the patches that are actually required, then delegates to the
standard ms-swift rlhf entry point (pip install ms-swift>=4.0).

Usage: invoked by train.sh via
    python3 run_rlhf_compat.py <all normal swift rlhf args>
"""
import runpy
import sys


def _patch_fsdp():
    """Alias FSDPModule -> FullyShardedDataParallel when the symbol is absent."""
    try:
        import torch.distributed.fsdp as fsdp
    except Exception:
        return
    if not hasattr(fsdp, "FSDPModule") and hasattr(fsdp, "FullyShardedDataParallel"):
        fsdp.FSDPModule = fsdp.FullyShardedDataParallel


def _patch_matplotlib():
    """Suppress matplotlib ImportError in swift's tb_utils plot_images."""
    try:
        import swift.utils.tb_utils as tb_utils
    except Exception:
        return
    _orig = tb_utils.plot_images

    def plot_images(*args, **kwargs):
        try:
            return _orig(*args, **kwargs)
        except ModuleNotFoundError as exc:
            if exc.name != "matplotlib":
                raise

    tb_utils.plot_images = plot_images


if __name__ == "__main__":
    _patch_fsdp()
    _patch_matplotlib()
    runpy.run_module("swift.cli.rlhf", run_name="__main__")
