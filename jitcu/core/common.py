import hashlib
import logging
import os
from pathlib import Path

from .. import env

logging.basicConfig(level=logging.INFO)


def _get_rank() -> str | None:
    # torchrun / most launchers export RANK; fall back to LOCAL_RANK so single-node
    # runs without a global RANK still get labeled. Env-var only — avoids importing
    # torch.distributed (jitcu is often loaded before dist init).
    return os.environ.get("RANK") or os.environ.get("LOCAL_RANK")


_RANK = _get_rank()
_RANK_FIELD = f" rank={_RANK}" if _RANK is not None else ""
_LOG_FMT = f"%(asctime)s - pid=%(process)d{_RANK_FIELD} - %(levelname)s - %(message)s"


def _per_process_log_name() -> str:
    pid = os.getpid()
    if _RANK is not None:
        return f"jitcu.rank{_RANK}.pid{pid}.log"
    return f"jitcu.pid{pid}.log"


class JITCULogger(logging.Logger):
    def __init__(self, name):
        super().__init__(name)
        self.setLevel(logging.INFO)

        os.makedirs(env.JITCU_WORKSPACE_DIR, exist_ok=True)

        # stderr
        self.addHandler(logging.StreamHandler())
        # combined log: shared across all processes. POSIX guarantees atomic
        # appends up to PIPE_BUF for O_APPEND on local FS, so single-line
        # records won't tear; on NFS the guarantee is weaker, in which case
        # consult the per-process log below.
        self.addHandler(
            logging.FileHandler(env.JITCU_WORKSPACE_DIR / "jitcu.log")
        )
        # per-process log: only this process writes here, so always clean.
        self.addHandler(
            logging.FileHandler(
                env.JITCU_WORKSPACE_DIR / _per_process_log_name()
            )
        )

        formatter = logging.Formatter(_LOG_FMT)
        for h in self.handlers:
            h.setFormatter(formatter)

    def info(self, msg, *args, **kwargs):
        super().info("jitcu: " + str(msg), *args, **kwargs)


logger = JITCULogger("jitcu")


def hash_files(file_paths: list[str | Path]) -> str:
    BLOCK_SIZE = 4096
    md5_hash = hashlib.md5()
    for file_path in file_paths:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(BLOCK_SIZE), b""):
                md5_hash.update(chunk)
    return md5_hash.hexdigest()
