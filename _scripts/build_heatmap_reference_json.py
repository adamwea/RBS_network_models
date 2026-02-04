"""Build a canonical JSON heatmap reference for a sensitivity-analysis run.

Does NOT rerun simulations. It scans the run's `permutations/` folder and reads
`network_data.npy` files to rebuild a machine-readable reference that can be used
for target selection and follow-on plotting.

Example:
  python _scripts/build_heatmap_reference_json.py \
    --run-dir /path/to/sensitivity_analysis/run0000 \
    --evol-params /path/to/evol_params.py \
    --out /path/to/run0000/heatmap_reference.json
"""

from __future__ import annotations

from pathlib import Path

# Allow running as a script without installing the package.
import sys as _sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

from RBS_network_models.utils.heatmap_reference import build_heatmap_reference, write_heatmap_reference_json


def _import_evol_params(evol_params_path: Path) -> dict:
    """Load evol_params.py without requiring NetPyNE/NEURON."""
    import importlib.util

    evol_params_path = evol_params_path.expanduser().resolve()
    spec = importlib.util.spec_from_file_location("evol_params", evol_params_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from {evol_params_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]

    for attr in ("params", "evol_params"):
        if hasattr(mod, attr):
            val = getattr(mod, attr)
            if isinstance(val, dict):
                return val
    raise AttributeError(f"No dict named 'params' (or 'evol_params') found in {evol_params_path}")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True, help="Sensitivity analysis run dir (contains permutations/)")
    ap.add_argument("--out", type=Path, required=True, help="Output JSON path")
    ap.add_argument(
        "--evol-params",
        type=Path,
        default=None,
        help="Optional path to evol_params.py (for stable param ordering)",
    )
    ap.add_argument(
        "--max-abs-level",
        type=int,
        default=None,
        help="Override the inferred max abs level (e.g. 5 to produce [-5..+5] + baseline)",
    )
    ap.add_argument(
        "--no-origin",
        action="store_true",
        help="Do not try to use/resolve run_dir/_origin as the baseline",
    )
    ap.add_argument(
        "--max-metric-samples",
        type=int,
        default=3,
        help="How many network_data.npy files to sample when discovering metric paths",
    )

    args = ap.parse_args()

    params = None
    if args.evol_params is not None:
        params = _import_evol_params(args.evol_params)

    ref = build_heatmap_reference(
        run_dir=args.run_dir,
        params=params,
        include_origin=not args.no_origin,
        max_metric_samples=int(args.max_metric_samples),
        max_abs_level=args.max_abs_level,
    )

    out = write_heatmap_reference_json(ref, args.out)
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
