#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path


SRC_ROOT = Path("/media/discover/WM/bridge_dataset/bridge_v2/raw/bridge_data_v2")
DST_ROOT = Path("/home/discover/sam3d_gs/raw_data")


def find_traj_groups(scene_dir: Path) -> dict[str, list[Path]]:
    task_groups: dict[str, list[Path]] = {}
    for path in scene_dir.rglob("traj_group*"):
        if not path.is_dir():
            continue
        rel_parts = path.relative_to(scene_dir).parts
        if not rel_parts:
            continue
        task_name = rel_parts[0]
        task_groups.setdefault(task_name, []).append(path)
    for task, groups in task_groups.items():
        task_groups[task] = sorted(groups, key=lambda p: p.as_posix())
    return task_groups


def iter_traj_dirs(traj_group_dir: Path) -> list[Path]:
    traj_dirs = [
        child
        for child in sorted(traj_group_dir.iterdir(), key=lambda p: p.name)
        if child.is_dir() and child.name.startswith("traj")
    ]
    return traj_dirs


def extract_index(name: str, prefix: str) -> int | None:
    if not name.startswith(prefix):
        return None
    tail = name[len(prefix) :]
    if tail.isdigit():
        return int(tail)
    match = re.match(r"(\d+)", tail)
    return int(match.group(1)) if match else None


def main() -> None:
    if not SRC_ROOT.exists():
        raise SystemExit(f"源路径不存在: {SRC_ROOT}")
    DST_ROOT.mkdir(parents=True, exist_ok=True)

    for scene_dir in sorted([p for p in SRC_ROOT.iterdir() if p.is_dir()], key=lambda p: p.name):
        task_groups = find_traj_groups(scene_dir)
        if not task_groups:
            print(f"[skip] 未找到 traj_group*: {scene_dir.name}")
            continue

        scene_dst = DST_ROOT / scene_dir.name
        scene_dst.mkdir(parents=True, exist_ok=True)

        mapping: dict[str, str] = {}

        for task_name in sorted(task_groups.keys()):
            groups = task_groups[task_name]
            group_counter = 0
            for group_dir in groups:
                traj_dirs = iter_traj_dirs(group_dir)
                if not traj_dirs:
                    continue

                out_task_name = task_name if group_counter == 0 else f"{task_name}{group_counter}"
                out_task_dir = scene_dst / out_task_name
                out_task_dir.mkdir(parents=True, exist_ok=True)
                group_index = extract_index(group_dir.name, "traj_group")
                if group_index is None:
                    group_index = group_counter
                group_counter += 1

                for traj_idx, traj_dir in enumerate(traj_dirs):
                    traj_index = extract_index(traj_dir.name, "traj")
                    if traj_index is None:
                        traj_index = traj_idx
                    dst_name = f"traj{group_index}_{traj_index}"
                    dst_traj_dir = out_task_dir / dst_name
                    shutil.copytree(traj_dir, dst_traj_dir)

                    src_rel = traj_dir.relative_to(SRC_ROOT).as_posix()
                    dst_rel = dst_traj_dir.relative_to(DST_ROOT).as_posix()
                    mapping[src_rel] = dst_rel

        if mapping:
            mapping_path = scene_dst / "path_map.json"
            with mapping_path.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "source_root": "bridge_data_v2",
                        "target_root": "raw_data",
                        "mappings": mapping,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            print(f"[ok] {scene_dir.name}: {len(mapping)} 条映射")
        else:
            print(f"[skip] {scene_dir.name} 未找到 traj* 目录")


if __name__ == "__main__":
    main()
