import importlib.util
import os
from dataclasses import dataclass, field
from pathlib import Path

from .common import logger


@dataclass(frozen=True)
class ExternalLib:
    name: str
    env_vars: list[str] = field(default_factory=list)
    default_paths: list[str] = field(default_factory=list)
    python_packages: list[str] = field(default_factory=list)
    include_subdirs: list[str] = field(default_factory=lambda: ["include"])
    lib_subdirs: list[str] = field(default_factory=lambda: ["lib", "lib64"])
    link_libs: list[str] = field(default_factory=list)
    extra_cuda_cflags: list[str] = field(default_factory=list)
    probe_header: str | None = None


@dataclass(frozen=True)
class ResolvedExternal:
    name: str
    root: Path
    include_paths: list[Path]
    lib_paths: list[Path]
    link_libs: list[str]
    extra_cuda_cflags: list[str]


def _python_package_paths(pkg: str) -> list[Path]:
    try:
        spec = importlib.util.find_spec(pkg)
    except (ImportError, ValueError):
        return []
    if spec is None or not spec.submodule_search_locations:
        return []
    return [Path(p) for p in spec.submodule_search_locations]


def _find_root(lib: ExternalLib, hint: str | Path | None) -> Path:
    candidates: list[tuple[str, Path]] = []
    if hint is not None:
        candidates.append(("user-supplied path", Path(hint)))
    for var in lib.env_vars:
        val = os.environ.get(var)
        if val:
            candidates.append((f"${var}", Path(val)))
    for pkg in lib.python_packages:
        for p in _python_package_paths(pkg):
            candidates.append((f"pip pkg {pkg}", p))
    for p in lib.default_paths:
        candidates.append(("default path", Path(p)))

    checked: list[str] = []
    for origin, root in candidates:
        checked.append(f"{origin}={root}")
        if not root.is_dir():
            continue
        include_dirs = [root / s for s in lib.include_subdirs if (root / s).is_dir()]
        if not include_dirs:
            continue
        if lib.probe_header is not None and not any(
            (d / lib.probe_header).exists() for d in include_dirs
        ):
            continue
        return root

    if checked:
        raise RuntimeError(
            f"Could not locate external library '{lib.name}'. Tried: "
            + ", ".join(checked)
        )
    hints = "/".join("$" + v for v in lib.env_vars) or "(no env vars configured)"
    raise RuntimeError(
        f"Could not locate external library '{lib.name}': no candidates — "
        f"pass a path or set {hints}"
    )


def _resolve_link_lib(lib_paths: list[Path], name: str) -> str:
    """Return the correct linker flag for `name` given the files present in lib_paths.

    Prefer `-l<name>` when `lib<name>.so` or `lib<name>.a` exists (typical
    system install). Fall back to `-l:lib<name>.so.<N>` when only a versioned
    SONAME file is present (this is the case for pip-installed NVIDIA packages
    that ship `libnvshmem_host.so.3` without the unversioned symlink).
    """
    for lp in lib_paths:
        if (lp / f"lib{name}.so").exists() or (lp / f"lib{name}.a").exists():
            return f"-l{name}"
    for lp in lib_paths:
        matches = sorted(lp.glob(f"lib{name}.so.*"))
        if matches:
            return f"-l:{matches[-1].name}"
    # nothing matched — best effort, let the linker error out with a clear message
    return f"-l{name}"


def resolve_external(lib: ExternalLib, hint: str | Path | None) -> ResolvedExternal:
    root = _find_root(lib, hint)
    include_paths = [root / s for s in lib.include_subdirs if (root / s).is_dir()]
    lib_paths = [root / s for s in lib.lib_subdirs if (root / s).is_dir()]
    link_flags = [_resolve_link_lib(lib_paths, name) for name in lib.link_libs]
    resolved = ResolvedExternal(
        name=lib.name,
        root=root,
        include_paths=include_paths,
        lib_paths=lib_paths,
        link_libs=link_flags,
        extra_cuda_cflags=list(lib.extra_cuda_cflags),
    )
    logger.info(
        f"Resolved external lib '{lib.name}' at {root} "
        f"(include={include_paths}, lib={lib_paths}, link={link_flags})"
    )
    return resolved


def resolve_externals(
    spec: dict[str, str | Path | None] | list[str] | None,
) -> list[ResolvedExternal]:
    if spec is None:
        return []
    if isinstance(spec, list):
        spec = {name: None for name in spec}

    out: list[ResolvedExternal] = []
    for name, hint in spec.items():
        if name not in EXTERNAL_LIBS:
            raise KeyError(
                f"Unknown external lib '{name}'. Known: {sorted(EXTERNAL_LIBS)}"
            )
        out.append(resolve_external(EXTERNAL_LIBS[name], hint))
    return out


EXTERNAL_LIBS: dict[str, ExternalLib] = {
    "nvshmem": ExternalLib(
        name="nvshmem",
        env_vars=["NVSHMEM_HOME", "NVSHMEM_DIR", "NVSHMEM_PREFIX"],
        default_paths=["/usr/local/nvshmem", "/opt/nvshmem"],
        python_packages=["nvidia.nvshmem"],
        link_libs=["nvshmem_host", "nvshmem_device"],
        extra_cuda_cflags=["-rdc=true"],
        probe_header="nvshmem.h",
    ),
}
