import hashlib
import logging
import os
from pathlib import Path
from typing import List, Union

from .. import env

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
