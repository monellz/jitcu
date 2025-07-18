import os
import pathlib


def _get_workspace_dir_name() -> pathlib.Path:
    # import re
    # import warnings
    # from torch.utils.cpp_extension import _get_cuda_arch_flags
    # try:
    #     with warnings.catch_warnings():
    #         # Ignore the warning for TORCH_CUDA_ARCH_LIST not set
    #         warnings.filterwarnings(
    #             "ignore", r".*TORCH_CUDA_ARCH_LIST.*", module="torch"
    #         )
    #         flags = _get_cuda_arch_flags()
    #     arch = "_".join(sorted(set(re.findall(r"compute_(\d+)", "".join(flags)))))
    # except Exception:
    #     arch = "noarch"
    # # e.g.: $HOME/.cache/jitcu/75_80_89_90/
    # return pathlib.Path.home() / ".cache" / "jitcu" / arch

    # Now we support cuda/ascend, so we don't want to specify the arch
    return pathlib.Path.home() / ".cache" / "jitcu"


# use pathlib
JITCU_WORKSPACE_DIR = _get_workspace_dir_name()
JITCU_JIT_DIR = JITCU_WORKSPACE_DIR / "cached_ops"
JITCU_GEN_SRC_DIR = JITCU_WORKSPACE_DIR / "generated"
_package_root = pathlib.Path(__file__).resolve().parents[0]
JITCU_INCLUDE_DIR = _package_root / "data" / "include"

# external environment variables
JITCU_NVCC_KEEP = os.environ.get("JITCU_NVCC_KEEP", "0") == "1"
JITCU_FORCE_RECOMPILE = os.environ.get("JITCU_FORCE_RECOMPILE", "0") == "1"
JITCU_VERBOSE = os.environ.get("JITCU_VERBOSE", "0") == "1"
JITCU_ENABLE_PROFILER = os.environ.get("JITCU_ENABLE_PROFILER", "0") == "1"
