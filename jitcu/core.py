import hashlib
import logging
import os
import subprocess
from pathlib import Path
from typing import List, Optional, Union

from . import env
from .library import Library

logging.basicConfig(level=logging.INFO)


class JITCULogger(logging.Logger):

    def __init__(self, name):
        super().__init__(name)
        self.setLevel(logging.INFO)
        self.addHandler(logging.StreamHandler())
        log_path = env.JITCU_WORKSPACE_DIR / "jitcu.log"
        if not os.path.exists(log_path):
            os.makedirs(env.JITCU_WORKSPACE_DIR, exist_ok=True)
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
    nvcc_keep: bool = False,
    force_recompile: bool = False,
    verbose: bool = False,
):

    if build_directory is None:
        build_directory = env.JITCU_JIT_DIR / name
    build_directory = Path(build_directory)
    os.makedirs(build_directory, exist_ok=True)

    # overwrite options
    nvcc_keep = env.JITCU_NVCC_KEEP or nvcc_keep
    force_recompile = env.JITCU_FORCE_RECOMPILE or force_recompile
    verbose = env.JITCU_VERBOSE or verbose
    enable_profiler = env.JITCU_ENABLE_PROFILER
    if enable_profiler:
        force_recompile = True
        logger.warning("Profiling is enabled, force recompilation.")

    # check sources
    if isinstance(sources, str):
        assert not os.path.exists(
            sources
        ), f"str-typed sources should not be a file path: {sources}"

        source_path = build_directory / f"{name}.cu"
        with open(source_path, "w") as f:
            f.write(sources)
            f.flush()
        sources = [source_path]
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

    # warn
    if "-DNDEBUG" not in extra_cflags and "-DNDEBUG" not in extra_cuda_cflags:
        # mostly for cute
        logger.warning(
            "It is recommended to use -DNDEBUG to avoid potential performance loss."
        )

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
        "--ptxas-options=--verbose,--register-usage-level=10,--warn-on-local-memory-usage",
    ]
    ldflags = []
    include_paths = [
        env.JITCU_INCLUDE_DIR,
    ]

    cflags += extra_cflags
    cuda_cflags += extra_cuda_cflags
    ldflags += extra_ldflags
    include_paths += extra_include_paths

    if nvcc_keep:
        # Still use --keep-dir to keep the object file
        # ref: https://github.com/NVIDIA/cutlass/blob/06e560d98a5fe8acb975db2c4c26817b6c90acb1/CMakeLists.txt#L444
        cuda_cflags.extend(["--keep", "--keep-dir", str(build_directory)])
    if verbose:
        cuda_cflags.extend(["-v"])
    if enable_profiler:
        cflags.extend(["-DJC_ENABLE_PROFILER"])

    logger.info(
        f"Loading... {name=} {func_names=} {func_params=} {sources=} {build_directory=}"
    )

    lib_name = f"{name}.so"
    lib_path = build_directory / lib_name

    # check if compilation is necessary
    lib_hash_path = build_directory / f"{name}.hash"
    need_recompile = True
    if not force_recompile and os.path.exists(lib_hash_path):
        hash_value = hash_files(file_paths=sources + [lib_path])
        with open(lib_hash_path, "r") as f:
            old_hash_value = f.read()
        if hash_value == old_hash_value:
            need_recompile = False
        else:
            logger.info(
                f"Trigger recompilation, hash_value is {hash_value} (prev: {old_hash_value})"
            )
            need_recompile = True

    if not need_recompile:
        logger.info(f"Using cached library: {lib_path}")
        return Library(
            lib_path=lib_path,
            func_names=func_names,
            func_params=func_params,
            device_type="cuda",
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
        *[str(s) for s in sources],
    ]

    logger.info(f"Compiling... {' '.join(cmd)}")

    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        raise RuntimeError(f"Failed to compile CUDA ops: {name}")

    if nvcc_keep:
        sass_path = build_directory / f"{name}.sass"
        o_path = build_directory / f"{name}.o"
        assert os.path.exists(o_path), f"object file does not exist: {o_path}"
        sass_cmd = [
            "cuobjdump",
            "--dump-sass",
            str(o_path),
        ]
        with open(sass_path, "w") as f:
            logger.info(f"Generating SASS (saved in {sass_path}): {' '.join(sass_cmd)}")
            ret = subprocess.run(sass_cmd, stdout=f, text=True)
            if ret.returncode != 0:
                raise RuntimeError(f"Failed to generate SASS: {name}")

    # save the hash value
    with open(lib_hash_path, "w") as f:
        f.write(hash_files(file_paths=sources + [lib_path]))

    return Library(
        lib_path=lib_path,
        func_names=func_names,
        func_params=func_params,
        device_type="cuda",
    )



def load_ascend_ops(
    name: str,
    sources: Union[List[Union[str, Path]], str],
    func_names: List[str],
    func_params: List[str],
    cce_aicore_arch: str,
    extra_cflags: Optional[List[str]] = None,
    extra_ldflags: Optional[List[str]] = None,
    extra_include_paths: Optional[List[Union[str, Path]]] = None,
    extra_library_paths: Optional[List[Union[str, Path]]] = None,
    build_directory: Optional[Union[str, Path]] = None,
    force_recompile: bool = False,
    verbose: bool = False,
):
    ASCEND_HOME_PATH = os.environ.get("ASCEND_HOME_PATH")
    assert ASCEND_HOME_PATH is not None, "ASCEND_HOME_PATH is not set"
    logger.info(f"ASCEND_HOME_PATH: {ASCEND_HOME_PATH}")

    if build_directory is None:
        build_directory = env.JITCU_JIT_DIR / name
    build_directory = Path(build_directory)
    os.makedirs(build_directory, exist_ok=True)

    # overwrite options
    force_recompile = env.JITCU_FORCE_RECOMPILE or force_recompile
    verbose = env.JITCU_VERBOSE or verbose
    enable_profiler = env.JITCU_ENABLE_PROFILER
    if enable_profiler:
        force_recompile = True
        logger.warning("Profiling is enabled, force recompilation.")

    # check sources
    if isinstance(sources, str):
        assert not os.path.exists(
            sources
        ), f"str-typed sources should not be a file path: {sources}"

        source_path = build_directory / f"{name}.cpp"
        with open(source_path, "w") as f:
            f.write(sources)
            f.flush()
        sources = [source_path]
    else:
        for path in sources:
            assert os.path.exists(path), f"source file does not exist: {path}"

    if extra_cflags is None:
        extra_cflags = []
    if extra_ldflags is None:
        extra_ldflags = []
    if extra_include_paths is None:
        extra_include_paths = []
    if extra_library_paths is None:
        extra_library_paths = []

    # warn
    if "-DNDEBUG" not in extra_cflags:
        # mostly for cute
        logger.warning(
            "It is recommended to use -DNDEBUG to avoid potential performance loss."
        )

    cflags = [
        "-O3",
        "-std=c++17",
    ]
    ldflags = []

    import platform
    machine = platform.machine()
    system = platform.system().lower()
    arch_os = f"{machine}-{system}"
    
    include_paths = [
        env.JITCU_INCLUDE_DIR,
        f"{ASCEND_HOME_PATH}/runtime/include",
        # f"{ASCEND_HOME_PATH}/{arch_os}/tikcpp/tikcfw/",
        f"{ASCEND_HOME_PATH}/{arch_os}/ascendc/include/basic_api",
        f"{ASCEND_HOME_PATH}/{arch_os}/ascendc/include/host_api",
        f"{ASCEND_HOME_PATH}/{arch_os}/ascendc/include/highlevel_api",
        f"{ASCEND_HOME_PATH}/{arch_os}/ascendc/include/basic_api/interface",
        f"{ASCEND_HOME_PATH}/{arch_os}/ascendc/include/basic_api/impl",
    ]
    library_paths = [
        f"{ASCEND_HOME_PATH}/runtime/lib64",
    ]

    cflags += extra_cflags
    ldflags += extra_ldflags
    include_paths += extra_include_paths
    library_paths += extra_library_paths

    if verbose:
        cflags.extend(["-v"])
    if enable_profiler:
        cflags.extend(["-DJC_ENABLE_PROFILER"])

    logger.info(
        f"Loading... {name=} {func_names=} {func_params=} {sources=} {build_directory=}"
    )

    lib_name = f"{name}.so"
    lib_path = build_directory / lib_name

    # check if compilation is necessary
    lib_hash_path = build_directory / f"{name}.hash"
    need_recompile = True
    if not force_recompile and os.path.exists(lib_hash_path):
        hash_value = hash_files(file_paths=sources + [lib_path])
        with open(lib_hash_path, "r") as f:
            old_hash_value = f.read()
        if hash_value == old_hash_value:
            need_recompile = False
        else:
            logger.info(
                f"Trigger recompilation, hash_value is {hash_value} (prev: {old_hash_value})"
            )
            need_recompile = True

    if not need_recompile:
        logger.info(f"Using cached library: {lib_path}")
        return Library(
            lib_path=lib_path,
            func_names=func_names,
            func_params=func_params,
            device_type="npu",
        )

    cmd = [
        "bisheng",
        f"--cce-aicore-arch={cce_aicore_arch}",
        "-x", "cce",
        *ldflags,
        *cflags,
        *["-I" + str(p) for p in include_paths],
        *["-L" + str(p) for p in library_paths],
        "-lascendcl",
        "-lruntime",
        "-fPIC",
        "--shared",
        "-o",
        str(lib_path),
        *[str(s) for s in sources],
    ]

    logger.info(f"Compiling... {' '.join(cmd)}")

    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        raise RuntimeError(f"Failed to compile Ascend ops: {name}")

    with open(lib_hash_path, "w") as f:
        f.write(hash_files(file_paths=sources + [lib_path]))

    return Library(
        lib_path=lib_path,
        func_names=func_names,
        func_params=func_params,
        device_type="npu",
    )
