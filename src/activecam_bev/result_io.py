from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


RESULT_META = {
    "use_camera": True,
    "use_lidar": False,
    "use_radar": False,
    "use_map": False,
    "use_external": False,
}


def write_result_json(
    path: Path, predictions: dict[str, list[dict[str, Any]]]
) -> str:
    """Write and hash a complete nuScenes-format result file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(
            {"meta": RESULT_META, "results": predictions},
            stream,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
