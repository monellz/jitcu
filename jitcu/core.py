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
    soc_version: str,
    extra_cflags: Optional[List[str]] = None,
    extra_ldflags: Optional[List[str]] = None,
    extra_include_paths: Optional[List[Union[str, Path]]] = None,
    build_directory: Optional[Union[str, Path]] = None,
    force_recompile: bool = False,
    use_cpu_mode: bool = False,
    verbose: bool = False,
):
    import platform

    machine = platform.machine()
    system = platform.system().lower()
    arch_os = f"{machine}-{system}"
    assert soc_version in ["Ascend910A"], f"Unsupported SOC version: {soc_version}"
    cce_aicore_arch_map = {
        "Ascend910A": "dav-c100",
    }
    cce_aicore_arch = cce_aicore_arch_map[soc_version]
    ASCEND_HOME_PATH = os.environ.get("ASCEND_HOME_PATH")
    assert ASCEND_HOME_PATH is not None, "ASCEND_HOME_PATH is not set"

    logger.info(f"ASCEND_HOME_PATH: {ASCEND_HOME_PATH}")
    logger.info(f"arch_os: {arch_os}")

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

    # warn
    if "-DNDEBUG" not in extra_cflags:
        # mostly for cute
        logger.warning(
            "It is recommended to use -DNDEBUG to avoid potential performance loss."
        )

    cflags = [
        "-g",
        "-std=c++17",
    ]
    ldflags = [
        f"-L{ASCEND_HOME_PATH}/runtime/lib64",
        "-lascendcl",
        "-lruntime",
    ]
    include_paths = [
        env.JITCU_INCLUDE_DIR,
        f"{ASCEND_HOME_PATH}/runtime/include",
        f"{ASCEND_HOME_PATH}/{arch_os}/ascendc/include/basic_api",
        f"{ASCEND_HOME_PATH}/{arch_os}/ascendc/include/host_api",
        f"{ASCEND_HOME_PATH}/{arch_os}/ascendc/include/highlevel_api",
        f"{ASCEND_HOME_PATH}/{arch_os}/ascendc/include/basic_api/interface",
        f"{ASCEND_HOME_PATH}/{arch_os}/ascendc/include/basic_api/impl",
    ]

    cflags += extra_cflags
    ldflags += extra_ldflags
    include_paths += extra_include_paths

    if use_cpu_mode:
        cflags.extend(["-O0", "-g"])
    else:
        cflags.extend(
            [
                "-O3",
                f"--cce-aicore-arch={cce_aicore_arch}",
                "-x",
                "cce",
            ]
        )

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

    if use_cpu_mode:
        # assert False, "CPU mode has bugs now"
        cmake_build_directory = build_directory / f"{name}_cmake_build"
        cmake_list_path = cmake_build_directory / "CMakeLists.txt"
        cmake_list_template = r"""
cmake_minimum_required(VERSION 3.10)
project(_TARGET_NAME_)
set(CMAKE_CXX_STANDARD 17)
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
find_package(tikicpulib REQUIRED)
add_library(_TARGET_NAME_ SHARED _KERNEL_FILES_)
set_target_properties(_TARGET_NAME_ PROPERTIES PREFIX "")
target_link_libraries(_TARGET_NAME_ PUBLIC tikicpulib::_SOC_VERSION_)
target_compile_options(_TARGET_NAME_ PRIVATE _C_FLAGS_)
target_include_directories(_TARGET_NAME_ PRIVATE _INCLUDE_PATHS_)
target_link_libraries(_TARGET_NAME_ PRIVATE _LD_FLAGS_)
install(TARGETS _TARGET_NAME_ LIBRARY DESTINATION .)
        """
        cmake_list_template = cmake_list_template.replace("_TARGET_NAME_", name)
        cmake_list_template = cmake_list_template.replace(
            "_KERNEL_FILES_", " ".join([str(Path(s).resolve()) for s in sources])
        )
        cmake_list_template = cmake_list_template.replace("_C_FLAGS_", " ".join(cflags))
        cmake_list_template = cmake_list_template.replace(
            "_INCLUDE_PATHS_", " ".join([str(Path(p).resolve()) for p in include_paths])
        )
        cmake_list_template = cmake_list_template.replace(
            "_LD_FLAGS_", " ".join(ldflags)
        )
        cmake_list_template = cmake_list_template.replace("_SOC_VERSION_", soc_version)

        os.makedirs(cmake_build_directory, exist_ok=True)
        with open(cmake_list_path, "w") as f:
            f.write(cmake_list_template)

        configure_cmd = [
            "cmake",
            "-DCMAKE_BUILD_TYPE=Debug",
            "-B",
            str(cmake_build_directory),
            "-S",
            str(cmake_build_directory),
            f"-DCMAKE_PREFIX_PATH={ASCEND_HOME_PATH}/tools/tikicpulib/lib/cmake",
        ]

        logger.info(f"Configuring CPU mode... {' '.join(configure_cmd)}")

        if subprocess.run(configure_cmd).returncode != 0:
            raise RuntimeError(f"Failed to build Ascend CPU mode ops: {name}")

        build_cmd = [
            "cmake",
            "--build",
            str(cmake_build_directory),
        ]
        logger.info(f"Building CPU mode... {' '.join(build_cmd)}")
        if subprocess.run(build_cmd).returncode != 0:
            raise RuntimeError(f"Failed to build Ascend CPU mode ops: {name}")

        install_cmd = [
            "cmake",
            "--install",
            str(cmake_build_directory),
            "--prefix",
            str(build_directory),
        ]
        logger.info(f"Installing CPU mode... {' '.join(install_cmd)}")
        if subprocess.run(install_cmd).returncode != 0:
            raise RuntimeError(f"Failed to install Ascend CPU mode ops: {name}")

        with open(lib_hash_path, "w") as f:
            f.write(hash_files(file_paths=sources + [lib_path]))

        return Library(
            lib_path=str(lib_path),
            func_names=func_names,
            func_params=func_params,
            device_type="npu",
        )
    else:
        cmd = [
            "bisheng",
            *cflags,
            *["-I" + str(p) for p in include_paths],
            *ldflags,
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
            lib_path=str(lib_path),
            func_names=func_names,
            func_params=func_params,
            device_type="npu",
        )
