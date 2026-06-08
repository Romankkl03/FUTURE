from __future__ import annotations

import sys
from pathlib import Path


def add_project_paths() -> None:
    for root in [Path.cwd(), *Path.cwd().parents]:
        src_dir = root / "src"
        if (root / "pyproject.toml").is_file() and src_dir.is_dir():
            for path in (str(root), str(src_dir)):
                if path not in sys.path:
                    sys.path.insert(0, path)
            return
    raise RuntimeError("Could not find project root with pyproject.toml and src/")


add_project_paths()

from experiments.fusion_over_raw_experiment import parse_args, run_all


if __name__ == "__main__":
    run_all(parse_args())
