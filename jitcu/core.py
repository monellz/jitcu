import os
import subprocess
import logging
from pathlib import Path
from typing import List, Optional, Union

from .env import JITCU_JIT_DIR, JITCU_INCLUDE_DIR
from .library import Library

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_cuda_ops(
  name: str,
  sources: List[Union[str, Path]],
  func_names: List[str],
  func_params: List[str],
  extra_cflags: Optional[List[str]] = None,
  extra_cuda_cflags: Optional[List[str]] = None,
  extra_ldflags: Optional[List[str]] = None,
  extra_include_paths: Optional[List[Union[str, Path]]] = None,
  build_directory: Optional[Union[str, Path]] = None,
):
  if extra_cflags is None:
    extra_cflags = []
  if extra_cuda_cflags is None:
    extra_cuda_cflags = []
  if extra_ldflags is None:
    extra_ldflags = []
  if extra_include_paths is None:
    extra_include_paths = []

  cflags = []
  cuda_cflags = [
    "-O3",
    "-std=c++17",
    "-use_fast_math",
  ]
  ldflags = []
  include_paths = [
    JITCU_INCLUDE_DIR,
  ]

  cflags += extra_cflags
  cuda_cflags += extra_cuda_cflags
  ldflags += extra_ldflags
  include_paths += extra_include_paths

  if build_directory is None:
    build_directory = JITCU_JIT_DIR / name
  
  build_directory = Path(build_directory)

  logger.info(f"Loading... {name=} {func_names=} {func_params=} {build_directory=}")
  os.makedirs(build_directory, exist_ok=True)

  lib_name = f"{name}.so"
  lib_path = build_directory / lib_name
  cmd = [
    "nvcc",
    "--compiler-options",
    "'-fPIC'",
    "-lineinfo",
    "--shared",
    "-o",
    str(lib_path),
    *cflags,
    *cuda_cflags,
    *ldflags,
    *["-I" + str(p) for p in include_paths],

    *sources,
  ]

  logger.info(f"Compiling... {" ".join(cmd)}")

  ret = subprocess.run(cmd)
  if ret.returncode != 0:
    raise RuntimeError(f"Failed to compile CUDA ops: {name}")

  return Library(
    lib_path=lib_path,
    func_names=func_names,
    func_params=func_params,
  )