#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import tarfile
import time
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence


CAMERAS = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)


def normalized_member_name(name: str) -> str:
    normalized = str(PurePosixPath(name.removeprefix("./")))
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe archive member: {name}")
    return normalized


def archive_layout(archive_root: Path, version: str) -> tuple[Path, list[Path]]:
    """Return the official metadata and blob archives for a dataset version."""
    if version == "v1.0-trainval":
        expected_blobs = 10
    elif version == "v1.0-test":
        expected_blobs = 1
    else:
        raise ValueError(f"unsupported nuScenes version: {version}")
    meta_candidates = [
        archive_root / f"{version}_meta.tgz",
        archive_root / f"{version}_meta.tar",
    ]
    meta_archive = next((path for path in meta_candidates if path.is_file()), meta_candidates[0])
    blob_archives = sorted(archive_root.glob(f"{version}*_blobs.tgz"))
    if not meta_archive.is_file() or len(blob_archives) != expected_blobs:
        raise FileNotFoundError(
            f"expected {version}_meta.tgz/.tar and {expected_blobs} blob archive(s) "
            f"under {archive_root}; found {len(blob_archives)}"
        )
    return meta_archive, blob_archives


def extract_metadata(meta_archive: Path, data_root: Path, version: str) -> None:
    marker = data_root / version / "sample.json"
    if marker.is_file():
        print(f"[META] already present: {marker}", flush=True)
        return
    data_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(meta_archive, "r:gz") as archive:
        for member in archive.getmembers():
            normalized_member_name(member.name)
        archive.extractall(data_root, filter="data")
    if not marker.is_file():
        raise RuntimeError(f"metadata extraction did not create {marker}")


def requested_camera_files(
    *,
    data_root: Path,
    version: str,
    split: str,
    cameras: Sequence[str] = CAMERAS,
) -> tuple[set[str], int, int]:
    from nuscenes import NuScenes
    from nuscenes.utils.splits import create_splits_scenes

    split_scenes = set(create_splits_scenes()[split])
    nusc = NuScenes(version=version, dataroot=str(data_root), verbose=False)
    requested: set[str] = set()
    sample_count = 0
    selected_scenes = 0
    for scene in nusc.scene:
        if scene["name"] not in split_scenes:
            continue
        selected_scenes += 1
        token = scene["first_sample_token"]
        while token:
            sample = nusc.get("sample", token)
            sample_count += 1
            for camera in cameras:
                sample_data = nusc.get("sample_data", sample["data"][camera])
                if not sample_data["is_key_frame"]:
                    raise RuntimeError(
                        f"sample {token} does not point to a keyframe for {camera}"
                    )
                requested.add(normalized_member_name(sample_data["filename"]))
            token = sample["next"]
    expected = sample_count * len(cameras)
    if len(requested) != expected:
        raise RuntimeError(
            f"expected {expected} unique camera files, found {len(requested)}"
        )
    return requested, sample_count, selected_scenes


def extract_selected_members(
    *,
    archives: Iterable[Path],
    requested: set[str],
    data_root: Path,
    min_free_gib: float,
) -> dict[str, object]:
    started = time.time()
    remaining = {
        name
        for name in requested
        if not (data_root / Path(*PurePosixPath(name).parts)).is_file()
    }
    reused = len(requested) - len(remaining)
    extracted = 0
    extracted_bytes = 0
    per_archive: dict[str, dict[str, int]] = {}
    for archive_path in archives:
        if not remaining:
            break
        archive_count = 0
        archive_bytes = 0
        print(
            f"[ARCHIVE] scanning {archive_path.name}; remaining={len(remaining)}",
            flush=True,
        )
        with tarfile.open(archive_path, "r|gz") as archive:
            for member in archive:
                name = normalized_member_name(member.name)
                if name not in remaining:
                    continue
                if not member.isfile():
                    raise RuntimeError(f"requested member is not a regular file: {name}")
                free_gib = shutil.disk_usage(data_root).free / (1024**3)
                if free_gib < min_free_gib:
                    raise RuntimeError(
                        f"free space {free_gib:.2f} GiB is below limit {min_free_gib:.2f} GiB"
                    )
                source = archive.extractfile(member)
                if source is None:
                    raise RuntimeError(f"cannot read archive member: {name}")
                target = data_root / Path(*PurePosixPath(name).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(target.suffix + ".partial")
                with source, temporary.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                os.replace(temporary, target)
                remaining.remove(name)
                extracted += 1
                archive_count += 1
                extracted_bytes += member.size
                archive_bytes += member.size
                if extracted % 500 == 0:
                    print(
                        f"[PROGRESS] extracted={extracted} remaining={len(remaining)} "
                        f"size_gib={extracted_bytes / (1024**3):.2f}",
                        flush=True,
                    )
        per_archive[archive_path.name] = {
            "files": archive_count,
            "bytes": archive_bytes,
        }
    return {
        "requested_files": len(requested),
        "reused_files": reused,
        "extracted_files": extracted,
        "extracted_bytes": extracted_bytes,
        "missing_files": sorted(remaining),
        "per_archive": per_archive,
        "elapsed_seconds": time.time() - started,
        "free_gib_after": shutil.disk_usage(data_root).free / (1024**3),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract only nuScenes keyframe camera images needed by a split."
    )
    parser.add_argument("--archive-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--min-free-gib", type=float, default=8.0)
    parser.add_argument("--manifest", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    archive_root = Path(args.archive_root).resolve()
    data_root = Path(args.data_root).resolve()
    manifest_path = Path(args.manifest).resolve()
    expected_split = "test" if args.version == "v1.0-test" else None
    if expected_split is not None and args.split != expected_split:
        raise ValueError(f"version {args.version} requires --split {expected_split}")
    if args.version == "v1.0-trainval" and args.split == "test":
        raise ValueError("v1.0-trainval does not contain the test split")
    meta_archive, blob_archives = archive_layout(archive_root, args.version)

    extract_metadata(meta_archive, data_root, args.version)
    requested, sample_count, scene_count = requested_camera_files(
        data_root=data_root,
        version=args.version,
        split=args.split,
    )
    print(
        f"[SPLIT] split={args.split} scenes={scene_count} samples={sample_count} "
        f"camera_files={len(requested)}",
        flush=True,
    )
    result = extract_selected_members(
        archives=blob_archives,
        requested=requested,
        data_root=data_root,
        min_free_gib=args.min_free_gib,
    )
    manifest = {
        "archive_root": str(archive_root),
        "data_root": str(data_root),
        "version": args.version,
        "split": args.split,
        "scene_count": scene_count,
        "sample_count": sample_count,
        "camera_count": len(CAMERAS),
        "cameras": list(CAMERAS),
        "result": result,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if result["missing_files"]:
        raise RuntimeError(
            f"{len(result['missing_files'])} requested camera files were not found"
        )
    print(
        f"[COMPLETE] files={len(requested)} "
        f"new_gib={result['extracted_bytes'] / (1024**3):.2f} "
        f"free_gib={result['free_gib_after']:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
