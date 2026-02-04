from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class PermutationCell:
    param: str
    level: int
    sim_label: str
    sim_dir: Path
    network_data_npy: Path
    data_pkl: Path | None
    cfg_json: Path | None


_LEVEL_RE = re.compile(r"^(?P<param>.*?)(?P<level>[+-]\d+)$")


def parse_permutation_label(label: str) -> tuple[str, int] | None:
    """Parse labels like 'E_diam_mean-1' or 'probEE+5' -> (param, level)."""
    m = _LEVEL_RE.match(label)
    if not m:
        return None
    param = m.group("param")
    level = int(m.group("level"))
    if level == 0:
        # Sensitivity runs typically omit 0; reserve 0 for baseline.
        return None
    return param, level


def _load_npy_dict(path: Path) -> dict[str, Any]:
    loaded = np.load(path, allow_pickle=True)
    if isinstance(loaded, np.ndarray) and loaded.shape == ():
        obj = loaded.item()
    else:
        obj = loaded
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict-like npy payload at {path}, got {type(obj)}")
    return obj


def discover_permutations(permutations_dir: Path) -> dict[str, dict[int, PermutationCell]]:
    """Return param -> level -> PermutationCell."""
    by_param: dict[str, dict[int, PermutationCell]] = {}

    for child in sorted(permutations_dir.iterdir()):
        if not child.is_dir():
            continue
        parsed = parse_permutation_label(child.name)
        if parsed is None:
            continue
        param, level = parsed

        network_data_npy = child / "network_data.npy"
        if not network_data_npy.exists():
            continue

        data_pkl: Path | None = None
        cfg_json: Path | None = None
        # Typical naming: <label>_data.pkl and <label>_cfg.json in the permutation directory.
        pkl_candidate = child / f"{child.name}_data.pkl"
        if pkl_candidate.exists():
            data_pkl = pkl_candidate
        cfg_candidate = child / f"{child.name}_cfg.json"
        if cfg_candidate.exists():
            cfg_json = cfg_candidate

        cell = PermutationCell(
            param=param,
            level=level,
            sim_label=child.name,
            sim_dir=child,
            network_data_npy=network_data_npy,
            data_pkl=data_pkl,
            cfg_json=cfg_json,
        )
        by_param.setdefault(param, {})[level] = cell

    return by_param


def resolve_origin_dir(run_dir: Path) -> tuple[Path | None, dict[str, Any]]:
    """Resolve run_dir/_origin if present. Does not raise on missing/broken links."""
    meta: dict[str, Any] = {
        "marker": str(run_dir / "_origin"),
        "exists": False,
        "resolved": None,
        "network_data_npy": None,
        "error": None,
    }

    marker = run_dir / "_origin"
    if not marker.exists() and not marker.is_symlink():
        meta["error"] = "_origin marker not found"
        return None, meta

    try:
        resolved = marker.resolve(strict=False)
        meta["resolved"] = str(resolved)
        if resolved.exists() and resolved.is_dir():
            meta["exists"] = True
            npy = resolved / "network_data.npy"
            meta["network_data_npy"] = str(npy)
            if npy.exists():
                return resolved, meta
            meta["error"] = "origin dir exists but network_data.npy missing"
            return resolved, meta
        meta["error"] = "origin target missing or not a directory"
        return resolved, meta
    except Exception as e:
        meta["error"] = f"failed to resolve _origin: {e}"
        return None, meta


def _is_numeric_key_dict(d: Any) -> bool:
    return isinstance(d, dict) and d and all(re.match(r"^\d+$", str(k)) for k in d)


_DEFAULT_KEYS_TO_IGNORE = {
    "std",
    "cov",
    "median",
    "burst_ids",
    "burst_part",
    "burst_parts",
    ".data",
    "unit_metrics",
    "gids",
    "spiking_metrics_by_unit",
    "spiking_times_by_unit",
    ".unit_metrics",
    "unit_types",
    "min",
    "max",
    "simData",
    "popData",
    "cellData",
}


def find_metric_paths_from_samples(
    npy_paths: Iterable[Path],
    *,
    max_samples: int = 3,
    parent_path: str | None = None,
    keys_to_ignore: set[str] | None = None,
    scalar_only: bool = True,
) -> list[str]:
    """Return sorted metric paths (dot-separated) discovered from a few network_data.npy samples."""
    ignore = _DEFAULT_KEYS_TO_IGNORE if keys_to_ignore is None else keys_to_ignore
    found: set[str] = set()

    def recurse(obj: Any, path: str | None) -> None:
        if isinstance(obj, dict):
            if _is_numeric_key_dict(obj):
                if path is not None and not any(ign in path for ign in ignore):
                    found.add(path)
                return
            for k, v in obj.items():
                next_path = f"{path}.{k}" if path else str(k)
                recurse(v, next_path)
        elif isinstance(obj, (int, float, np.number)):
            if path is not None and not any(ign in path for ign in ignore):
                found.add(path)
        elif isinstance(obj, list):
            if not obj:
                return
            if all(isinstance(x, (int, float, np.number)) for x in obj):
                if scalar_only:
                    return
                if path is not None and not any(ign in path for ign in ignore):
                    found.add(path)

    for i, p in enumerate(npy_paths):
        if i >= max_samples:
            break
        try:
            obj = _load_npy_dict(p)
            recurse(obj, parent_path)
        except Exception:
            continue

    return sorted(found)


def extract_metric(obj: dict[str, Any], metric_path: str) -> float | None:
    """Extract a scalar metric at metric_path from a loaded network_data dict."""
    cur: Any = obj
    for part in metric_path.split("."):
        if not isinstance(cur, dict):
            return None
        if part not in cur:
            return None
        cur = cur[part]

    if isinstance(cur, (int, float, np.number)):
        val = float(cur)
        if np.isfinite(val):
            return val
        return None

    return None


def metric_paths_from_heatmap_pngs(run_dir: Path) -> list[str] | None:
    """Infer metric paths from existing heatmap PNG filenames.

    Expects filenames like `heatmap_hyperburst_metrics.burst_metrics.burst_rate.png`.
    Returns the inferred metric paths (without the `heatmap_` prefix), or None if
    there is no heatmaps directory.
    """
    heatmaps_dir = run_dir / "heatmaps"
    if not heatmaps_dir.exists():
        return None
    paths: list[str] = []
    for p in sorted(heatmaps_dir.glob("heatmap_*.png")):
        name = p.name
        if not name.startswith("heatmap_") or not name.endswith(".png"):
            continue
        metric = name[len("heatmap_") : -len(".png")]
        if metric and metric not in paths:
            paths.append(metric)
    return paths or None


def build_heatmap_reference(
    *,
    run_dir: Path,
    params: dict[str, Any] | None = None,
    include_origin: bool = True,
    max_metric_samples: int = 3,
    max_abs_level: int | None = None,
    metric_paths: list[str] | None = None,
) -> dict[str, Any]:
    """Build a canonical heatmap reference JSON payload without rerunning simulations."""
    run_dir = run_dir.resolve()
    permutations_dir = run_dir / "permutations"
    if not permutations_dir.exists():
        raise FileNotFoundError(f"Missing permutations dir: {permutations_dir}")

    by_param = discover_permutations(permutations_dir)
    if not by_param:
        raise ValueError(f"No permutation cells found under {permutations_dir}")

    # Determine parameter order
    if params is None:
        param_list = sorted(by_param.keys())
    else:
        # keep only params that actually appear
        param_list = [k for k in sorted(params.keys()) if k in by_param]
        if not param_list:
            param_list = sorted(by_param.keys())

    # Determine level axis
    all_levels = [lvl for cells in by_param.values() for lvl in cells.keys()]
    inferred_max_abs = int(max(abs(int(x)) for x in all_levels))
    if max_abs_level is None:
        max_abs_level = inferred_max_abs
    max_abs_level = int(max_abs_level)

    levels_axis = list(range(-max_abs_level, 0)) + [0] + list(range(1, max_abs_level + 1))

    # Origin (baseline)
    origin_dir: Path | None = None
    origin_meta: dict[str, Any] = {
        "marker": str(run_dir / "_origin"),
        "exists": False,
        "resolved": None,
        "network_data_npy": None,
        "error": "origin not requested",
    }
    if include_origin:
        origin_dir, origin_meta = resolve_origin_dir(run_dir)

    # Metric paths: prefer existing heatmap PNGs (keeps JSON aligned with what you plotted).
    if metric_paths is None:
        metric_paths = metric_paths_from_heatmap_pngs(run_dir)

    if metric_paths is None:
        sample_paths: list[Path] = []
        if origin_dir is not None and (origin_dir / "network_data.npy").exists():
            sample_paths.append(origin_dir / "network_data.npy")
        # add a few permutation paths
        for p in param_list:
            for lvl in sorted(by_param[p].keys(), key=lambda x: abs(x)):
                sample_paths.append(by_param[p][lvl].network_data_npy)
                if len(sample_paths) >= max_metric_samples:
                    break
            if len(sample_paths) >= max_metric_samples:
                break
        metric_paths = find_metric_paths_from_samples(
            sample_paths,
            max_samples=max_metric_samples,
            parent_path=None,
            scalar_only=True,
        )

    # Build cell grid
    cells_grid: list[list[dict[str, Any]]] = []
    cell_id_to_pos: dict[str, list[int]] = {}

    for row_i, p in enumerate(param_list):
        row: list[dict[str, Any]] = []
        for col_i, lvl in enumerate(levels_axis):
            cell_id = f"{p}:{lvl:+d}" if lvl != 0 else f"{p}:baseline"

            if lvl == 0:
                sim_dir = str(origin_dir) if origin_dir is not None else None
                network_data_npy = str((origin_dir / "network_data.npy") if origin_dir is not None else None)
                data_pkl = None
                cfg_json = None
                exists = {
                    "sim_dir": bool(origin_dir is not None and origin_dir.exists()),
                    "network_data_npy": bool(origin_dir is not None and (origin_dir / "network_data.npy").exists()),
                }
            else:
                cell = by_param.get(p, {}).get(int(lvl), None)
                sim_dir = str(cell.sim_dir) if cell is not None else None
                network_data_npy = str(cell.network_data_npy) if cell is not None else None
                data_pkl = str(cell.data_pkl) if (cell is not None and cell.data_pkl is not None) else None
                cfg_json = str(cell.cfg_json) if (cell is not None and cell.cfg_json is not None) else None
                exists = {
                    "sim_dir": bool(cell is not None and cell.sim_dir.exists()),
                    "network_data_npy": bool(cell is not None and cell.network_data_npy.exists()),
                    "data_pkl": bool(cell is not None and cell.data_pkl is not None and cell.data_pkl.exists()),
                    "cfg_json": bool(cell is not None and cell.cfg_json is not None and cell.cfg_json.exists()),
                }

            row.append(
                {
                    "cell_id": cell_id,
                    "param": p,
                    "level": int(lvl),
                    "sim_dir": sim_dir,
                    "network_data_npy": network_data_npy,
                    "data_pkl": data_pkl,
                    "cfg_json": cfg_json,
                    "exists": exists,
                }
            )
            cell_id_to_pos[cell_id] = [row_i, col_i]
        cells_grid.append(row)

    # Build metric grids (values only; join to sim paths via cells_grid indices)
    metrics_out: dict[str, Any] = {
        mp: {"values": [[None for _ in levels_axis] for _ in param_list]} for mp in metric_paths
    }

    metric_parts: list[tuple[str, list[str]]] = [(mp, mp.split(".")) for mp in metric_paths]

    def extract_metric_parts(obj: dict[str, Any], parts: list[str]) -> float | None:
        cur: Any = obj
        for part in parts:
            if not isinstance(cur, dict) or part not in cur:
                return None
            cur = cur[part]
        if isinstance(cur, (int, float, np.number)):
            v = float(cur)
            return v if np.isfinite(v) else None
        return None

    # origin baseline column (lvl==0)
    origin_obj: dict[str, Any] | None = None
    if origin_dir is not None:
        npy = origin_dir / "network_data.npy"
        if npy.exists():
            try:
                origin_obj = _load_npy_dict(npy)
            except Exception:
                origin_obj = None

    for row_i, p in enumerate(param_list):
        for col_i, lvl in enumerate(levels_axis):
            obj: dict[str, Any] | None
            if lvl == 0:
                obj = origin_obj
            else:
                cell = by_param.get(p, {}).get(int(lvl))
                if cell is None or not cell.network_data_npy.exists():
                    obj = None
                else:
                    try:
                        obj = _load_npy_dict(cell.network_data_npy)
                    except Exception:
                        obj = None

            if obj is None:
                continue

            for mp, parts in metric_parts:
                metrics_out[mp]["values"][row_i][col_i] = extract_metric_parts(obj, parts)

    ref: dict[str, Any] = {
        "schema_version": "heatmap_reference_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "permutations_dir": str(permutations_dir),
        "origin": origin_meta,
        "axes": {
            "params": param_list,
            "levels": levels_axis,
        },
        "cells": cells_grid,
        "metrics": metrics_out,
        "index": {
            "cell_id_to_pos": cell_id_to_pos,
        },
        "notes": {
            "levels_axis": "levels are integer perturbation steps; 0 is reserved for baseline/origin",
            "metric_values": "metrics.values is aligned to axes.params (rows) and axes.levels (cols)",
        },
    }
    return ref


def write_heatmap_reference_json(ref: dict[str, Any], out_path: Path) -> Path:
    out_path = out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(ref, f, indent=2, sort_keys=False)
    return out_path
