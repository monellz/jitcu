import hashlib
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Union

from .env import JITCU_INCLUDE_DIR, JITCU_JIT_DIR, JITCU_WORKSPACE_DIR
from .library import Library

logging.basicConfig(level=logging.INFO)


class JITCULogger(logging.Logger):

    def __init__(self, name):
        super().__init__(name)
        self.setLevel(logging.INFO)
        self.addHandler(logging.StreamHandler())
        log_path = JITCU_WORKSPACE_DIR / "jitcu.log"
        if not os.path.exists(log_path):
            os.makedirs(JITCU_WORKSPACE_DIR, exist_ok=True)
            # create an empty file
            with open(log_path, "w") as f:  # noqa: F841
                pass
        self.addHandler(logging.FileHandler(log_path))
        # set the format of the log
        self.handlers[0].setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        self.handlers[1].setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )

    def info(self, msg):
        super().info("jitcu: " + msg)


logger = JITCULogger("jitcu")


def hash_files(file_paths: List[Union[str, Path]]) -> str:
    BLOCK_SIZE = 4096
    md5_hash = hashlib.md5()
    for file_path in file_paths:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(BLOCK_SIZE), b""):
                md5_hash.update(chunk)
    return md5_hash.hexdigest()


def load_cuda_ops(
    name: str,
    sources: Union[List[Union[str, Path]], str],
    func_names: List[str],
    func_params: List[str],
    arches: Optional[List[str]] = None,
    extra_cflags: Optional[List[str]] = None,
    extra_cuda_cflags: Optional[List[str]] = None,
    extra_ldflags: Optional[List[str]] = None,
    extra_include_paths: Optional[List[Union[str, Path]]] = None,
    build_directory: Optional[Union[str, Path]] = None,
    keep_intermediates: bool = True,
    force_recompile: bool = False,
):
    # check sources
    if isinstance(sources, str):
        assert not os.path.exists(
            sources
        ), f"str-typed sources should not be a file path: {sources}"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cu", delete=False) as f:
            f.write(sources)
            f.flush()
        sources = [f.name]
    else:
        for path in sources:
            assert os.path.exists(path), f"source file does not exist: {path}"

    if extra_cflags is None:
        extra_cflags = []
    if extra_cuda_cflags is None:
        extra_cuda_cflags = []
    if extra_ldflags is None:
        extra_ldflags = []
    if extra_include_paths is None:
        extra_include_paths = []

    arch_flags = []
    if arches is not None and len(arches) > 0:
        for arch in arches:
            arch_flags.append("-gencode")
            arch_flags.append(f"arch=compute_{arch},code=sm_{arch}")

    cflags = []
    cuda_cflags = [
        "-O3",
        "-std=c++17",
        "-use_fast_math",
        "--expt-relaxed-constexpr",
        "--ptxas-options=-v",
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

    if keep_intermediates:
        cuda_cflags.extend(["--keep", "--keep-dir", str(build_directory)])

    logger.info(
        f"Loading... {name=} {func_names=} {func_params=} {sources=} {build_directory=}"
    )
    os.makedirs(build_directory, exist_ok=True)

    lib_name = f"{name}.so"
    lib_path = build_directory / lib_name

    # check if compilation is necessary
    lib_hash_path = build_directory / f"{name}.hash"
    need_recompile = force_recompile
    if not need_recompile and os.path.exists(lib_hash_path):
        hash_value = hash_files(
            file_paths=sources + [lib_path] if os.path.exists(lib_path) else sources
        )
        with open(lib_hash_path, "r") as f:
            old_hash_value = f.read()
        if hash_value == old_hash_value:
            need_recompile = False

    if not need_recompile:
        logger.info(f"Using cached library: {lib_path}")
        return Library(
            lib_path=lib_path,
            func_names=func_names,
            func_params=func_params,
        )

    cmd = [
        "nvcc",
        *arch_flags,
        *cuda_cflags,
        *ldflags,
        *["-I" + str(p) for p in include_paths],
        "--compiler-options",
        "'-fPIC'",
        "-lineinfo",
        "--shared",
        *cflags,
        "-o",
        str(lib_path),
        *sources,
    ]

    logger.info(f"Compiling... {' '.join(cmd)}")

    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        raise RuntimeError(f"Failed to compile CUDA ops: {name}")

    # save the hash value
    with open(lib_hash_path, "w") as f:
        f.write(hash_files(file_paths=sources + [lib_path]))

    return Library(
        lib_path=lib_path,
        func_names=func_names,
        func_params=func_params,
    )
