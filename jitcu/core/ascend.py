import os
import platform
import subprocess
from pathlib import Path

from filelock import FileLock

from .. import env
from ..library import Library
from .common import hash_files, logger


def load_ascend_ops(
    name: str,
    sources: list[str | Path] | str,
    func_specs: dict[str, str],
    soc_version: str | None = None,
    extra_cflags: list[str] | None = None,
    extra_ldflags: list[str] | None = None,
    extra_include_paths: list[str | Path] | None = None,
    build_directory: str | Path | None = None,
    force_recompile: bool = False,
    verbose: bool = False,
):
    machine = platform.machine()
    system = platform.system().lower()
    arch_os = f"{machine}-{system}"
    cce_aicore_arch_map = {
        # ref: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_10053.html
        "Ascend910A": "dav-1001",
        "Ascend910B": "dav-2201",
        "Ascend950PR": "dav-3510",
    }
    if soc_version is None:
        import acl
        soc_name = acl.get_soc_name()
        soc_version = soc_name.split("_")[0]
        assert soc_version in cce_aicore_arch_map, f"Unsupported SOC version: {soc_version}({soc_name})"
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

    # check sources (str-source contents are written inside the build lock below
    # to avoid torn reads when multiple processes share the same build dir)
    pending_str_source: str | None = None
    if isinstance(sources, str):
        assert not os.path.exists(sources), (
            f"str-typed sources should not be a file path: {sources}"
        )
        source_path = build_directory / f"{name}.cpp"
        pending_str_source = sources
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
        "-O3",
        "-fPIC",
        "-shared",
        # ascend related
        "-Wno-macro-redefined",
        "-Wno-ignored-attributes",
        "-xcce",
        f"--npu-arch={cce_aicore_arch}",
        "-mllvm", "-cce-aicore-stack-size=0x8000",
        "-mllvm", "-cce-aicore-function-stack-size=0x8000",
        "-mllvm", "-cce-aicore-record-overflow=true",
        "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false",
    ]
    ldflags = [
        f"-L{ASCEND_HOME_PATH}/runtime/lib64",
        "-lascendcl",
        "-lruntime",
    ]
    ascendc_include = f"{ASCEND_HOME_PATH}/{arch_os}/ascendc/include"
    include_paths: list[str | Path] = [
        env.JITCU_INCLUDE_DIR,
        # acl/acl.h and the rest of the host runtime headers
        f"{ASCEND_HOME_PATH}/{arch_os}/include",
        # AscendC kernel-side headers (kernel_operator.h + its interface/impl tree)
        f"{ascendc_include}/basic_api",
        f"{ascendc_include}/basic_api/interface",
        f"{ascendc_include}/basic_api/impl",
        f"{ascendc_include}/highlevel_api",
        # some AscendC headers use root-relative includes ("include/utils/...")
        # that resolve against the `asc` tree.
        f"{ASCEND_HOME_PATH}/{arch_os}/asc",
    ]

    cflags += extra_cflags
    ldflags += extra_ldflags
    include_paths += extra_include_paths

    if verbose:
        cflags.extend(["-v"])
    if enable_profiler:
        cflags.extend(["-DJC_ENABLE_PROFILER"])

    logger.info(
        f"Loading... {name=} {func_specs=} {sources=} {build_directory=}"
    )

    lib_name = f"{name}.so"
    lib_path = build_directory / lib_name
    lib_hash_path = build_directory / f"{name}.hash"
    lock_path = build_directory / f"{name}.lock"

    # Serialize source-write / hash-check / build / hash-save across processes
    # sharing this build_directory. Lock is per-`name`, so different ops still
    # build in parallel.
    with FileLock(str(lock_path)):
        if pending_str_source is not None:
            with open(sources[0], "w") as f:
                f.write(pending_str_source)
                f.flush()

        # check if compilation is necessary
        need_recompile = True
        if not force_recompile and os.path.exists(lib_hash_path):
            hash_value = hash_files(file_paths=sources + [lib_path])
            with open(lib_hash_path) as f:
                old_hash_value = f.read()
            if hash_value == old_hash_value:
                need_recompile = False
            else:
                logger.info(
                    f"Trigger recompilation, hash {hash_value} (prev {old_hash_value})"
                )
                need_recompile = True

        if not need_recompile:
            logger.info(f"Using cached library: {lib_path}")
        else:
            cmd = [
                "bisheng",
                *cflags,
                *["-I" + str(p) for p in include_paths],
                *ldflags,
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
        func_specs=func_specs,
        device_type="npu",
    )
