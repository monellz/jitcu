import os
import platform
import subprocess
from pathlib import Path

from filelock import FileLock

from .. import env
from ..library import Library
from .common import hash_files, logger
from .externals import resolve_externals


def load_ascend_ops(
    name: str,
    sources: list[str | Path] | str,
    func_specs: dict[str, str],
    soc_version: str | None = None,
    external_libs: dict[str, str | Path | None] | list[str] | None = None,
    extra_cflags: list[str] | None = None,
    extra_ldflags: list[str] | None = None,
    extra_include_paths: list[str | Path] | None = None,
    extra_hash_files: list[str | Path] | None = None,
    build_directory: str | Path | None = None,
    force_recompile: bool = False,
    verbose: bool = False,
):
    machine = platform.machine()
    system = platform.system().lower()
    arch_os = f"{machine}-{system}"
    # CCE-front-end aicore arch (-xcce --cce-aicore-arch=...): needed for PTO-ISA / mixed
    # AIC+AIV kernels (SYNCALL etc.). ref:
    # https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_10053.html
    cce_aicore_arch_map = {
        "Ascend910A": "dav-c100",
        "Ascend910B": "dav-c220",
        "Ascend950PR": "dav-c310",
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

    # extra files (e.g. headers the sources #include) folded into the cache key so
    # editing them triggers a recompile even though they are not compiled directly.
    if extra_hash_files is None:
        extra_hash_files = []
    for path in extra_hash_files:
        assert os.path.exists(path), f"extra hash file does not exist: {path}"

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
        # ascend related (CCE front-end, for PTO-ISA / mixed AIC+AIV kernels)
        "-Wno-macro-redefined",
        "-Wno-ignored-attributes",
        "-xcce",
        f"--cce-aicore-arch={cce_aicore_arch}",
        "-mllvm", "-cce-aicore-addr-transform",
        "-mllvm", "-cce-aicore-stack-size=0x8000",
        "-mllvm", "-cce-aicore-function-stack-size=0x8000",
        "-mllvm", "-cce-aicore-record-overflow=true",
        "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false",
    ]
    ldflags = [
        # f"-L{ASCEND_HOME_PATH}/runtime/lib64",
        # "-lascendcl",
        # "-lruntime",
    ]
    include_paths: list[str | Path] = [
        env.JITCU_INCLUDE_DIR,
    ]
    # External libs (e.g. {"pto": "/path/to/pto-isa"}) are resolved and their includes
    # placed BEFORE {ASCEND}/include, so e.g. pto-isa's headers win over the same-named
    # `pto/` tree CANN ships under {ASCEND}/include.
    for resolved in resolve_externals(external_libs):
        include_paths += resolved.include_paths
        for lp in resolved.lib_paths:
            ldflags.append(f"-L{lp}")
            ldflags.append(f"-Wl,-rpath,{lp}")
        ldflags += resolved.link_libs
    # acl.h + host runtime headers — added AFTER the external includes (so they can't be
    # shadowed by CANN's pto/ here). The rest of the CANN device/AscendC trees are auto-
    # added by the CCE front-end.
    include_paths.append(f"{ASCEND_HOME_PATH}/include")

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
    hash_paths = [*sources, *extra_hash_files, lib_path]

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
            hash_value = hash_files(file_paths=hash_paths)
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
                f.write(hash_files(file_paths=hash_paths))

    return Library(
        lib_path=str(lib_path),
        func_specs=func_specs,
        device_type="npu",
    )
