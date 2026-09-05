"""The ``trajbench`` command line interface.

The CLI keeps imports lazy so validating configs and launching the CPU Tiny
runner work in installations without dashboard, Parquet, or FAISS extras.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from trajsimbench.analysis.figures import generate_figures
from trajsimbench.analysis.tables import generate_tables
from trajsimbench.orchestration.runner import (
    resolve_config,
    run_experiment,
)
from trajsimbench.storage.artifacts import validate_run_directory

KNOWN_METHODS = ("euclidean", "dtw", "hausdorff", "discrete_frechet", "lcss", "edr", "erp")
KNOWN_DATASETS = ("synthetic", "porto", "geolife", "tdrive", "ais", "germany")


def _json_print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def command_config_validate(path: str) -> int:
    try:
        try:
            from trajsimbench.config import load_config as strict_load_config
        except ImportError:
            config = resolve_config(path)
            resolved_hash = config["resolved_config_hash"]
        else:
            resolved = strict_load_config(path)
            config = resolved.to_dict()
            resolved_hash = resolved.config_hash
    except Exception as exc:
        print(f"invalid configuration: {exc}", file=sys.stderr)
        return 2
    _json_print(
        {
            "valid": True,
            "experiment_id": config["experiment_id"],
            "resolved_config_hash": resolved_hash,
        }
    )
    return 0


def command_config_resolve(path: str) -> int:
    try:
        try:
            from trajsimbench.config import load_config as strict_load_config
        except ImportError:
            config = resolve_config(path)
        else:
            config = strict_load_config(path).to_dict()
    except Exception as exc:
        print(f"could not resolve configuration: {exc}", file=sys.stderr)
        return 2
    _json_print(config)
    return 0


def command_run(
    path: str, *, resume: bool = False, dry_run: bool = False, force_stage: str | None = None
) -> int:
    try:
        result = run_experiment(path, resume=resume, dry_run=dry_run, force_stage=force_stage)
    except Exception as exc:
        print(
            f"run failed before artifact finalization: {type(exc).__name__}: {exc}", file=sys.stderr
        )
        return 1
    if isinstance(result, dict):
        _json_print(result)
        return 0
    _json_print(
        {
            "run_dir": str(result.run_dir),
            "run_id": result.run_id,
            "experiment_id": result.experiment_id,
            "status": result.status,
        }
    )
    return 0 if result.status == "complete" else 1


def command_analyze(experiment: str, results_root: str, output: str) -> int:
    root = (
        Path(results_root) / experiment
        if (Path(results_root) / experiment).exists()
        else Path(results_root)
    )
    out = Path(output)
    tables = generate_tables(root, out / "tables")
    figures = generate_figures(root, out / "figures")
    _json_print(
        {
            "tables": {key: str(value) for key, value in tables.items()},
            "figures": {key: str(value) for key, value in figures.items()},
        }
    )
    return 0


def command_validate_data(path: str) -> int:
    root = Path(path)
    if not root.exists():
        message = (
            f"data path does not exist: {root}. Prepare the dataset or pass a "
            "processed version directory."
        )
        print(
            message,
            file=sys.stderr,
        )
        return 2
    try:
        from trajsimbench.data.validation import validate_processed_dataset
    except ImportError:
        required = [
            root / "points.npy",
            root / "offsets.npy",
            root / "metadata.parquet",
            root / "dataset.json",
        ]
        missing = [str(item) for item in required if not item.exists()]
        if missing:
            print(
                "data validator unavailable and canonical files are missing: " + ", ".join(missing),
                file=sys.stderr,
            )
            return 2
        _json_print({"valid": True, "path": str(root), "validator": "minimal-contract"})
        return 0
    try:
        report = validate_processed_dataset(root)
    except Exception as exc:
        print(f"invalid data: {exc}", file=sys.stderr)
        return 2
    _json_print(report)
    return 0 if report.get("valid", False) else 2


def command_perturb(
    dataset: str, trajectory: str, transform: str, severity: float, seed: int, output: str
) -> int:
    """Apply one seeded transformation and write points plus provenance."""
    import numpy as np

    try:
        from trajsimbench.data.dataset import TrajectoryDataset
        from trajsimbench.perturbations import get_perturbation
    except ImportError as exc:
        print(f"perturbation support is unavailable: {exc}", file=sys.stderr)
        return 2

    source_path = Path(trajectory)
    if source_path.exists():
        try:
            source: Any = np.load(source_path, allow_pickle=False)
        except Exception as exc:
            print(f"could not read trajectory array {source_path}: {exc}", file=sys.stderr)
            return 2
    else:
        dataset_path = Path(dataset)
        if not dataset_path.exists():
            dataset_path = Path("data/processed") / dataset / "v1"
        if not dataset_path.exists():
            print(
                f"dataset path does not exist: {dataset_path}; prepare it or pass a .npy "
                "trajectory",
                file=sys.stderr,
            )
            return 2
        try:
            source = TrajectoryDataset.open(dataset_path).by_id(trajectory)
        except Exception as exc:
            print(f"could not resolve trajectory {trajectory!r}: {exc}", file=sys.stderr)
            return 2
    try:
        result = get_perturbation(transform).apply(source, severity=severity, seed=seed)
    except Exception as exc:
        print(f"perturbation failed: {exc}", file=sys.stderr)
        return 2
    if not result.generated or result.points is None:
        print(f"perturbation was not generated: {result.reason}", file=sys.stderr)
        return 1
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, result.points, allow_pickle=False)
    provenance_path = output_path.with_suffix(output_path.suffix + ".provenance.json")
    provenance_path.write_text(
        json.dumps(result.provenance.to_dict(), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    _json_print(
        {
            "dataset": dataset,
            "trajectory": trajectory,
            "transform": transform,
            "severity": severity,
            "seed": seed,
            "output": str(output_path),
            "provenance": str(provenance_path),
            "variant_id": result.variant_id,
            "output_hash": result.output_hash,
        }
    )
    return 0


def command_list(kind: str) -> int:
    _json_print(list(KNOWN_METHODS if kind == "methods" else KNOWN_DATASETS))
    return 0


def command_results_validate(run_dir: str) -> int:
    report = validate_run_directory(Path(run_dir))
    _json_print(report)
    return 0 if report["valid"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trajbench", description="TrajSimBench reproducible trajectory benchmark"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="prepare a configured dataset")
    prepare.add_argument("--dataset", required=True)
    prepare.add_argument("--config", required=True)
    prepare.add_argument("--dry-run", action="store_true")
    validate = subparsers.add_parser("validate-data")
    validate.add_argument("path")
    perturb = subparsers.add_parser("perturb")
    perturb.add_argument("--dataset", required=True)
    perturb.add_argument("--trajectory", required=True)
    perturb.add_argument("--type", dest="transform", required=True)
    perturb.add_argument("--severity", required=True, type=float)
    perturb.add_argument("--seed", required=True, type=int)
    perturb.add_argument("--output", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("config")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--force-stage")
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--experiment", required=True)
    analyze.add_argument("--results-root", default="results")
    analyze.add_argument("--output", required=True)
    dashboard = subparsers.add_parser("dashboard")
    dashboard.add_argument("--results-root", default="results")
    config = subparsers.add_parser("config")
    config_sub = config.add_subparsers(dest="config_command", required=True)
    cv = config_sub.add_parser("validate")
    cv.add_argument("file")
    cr = config_sub.add_parser("resolve")
    cr.add_argument("file")
    methods = subparsers.add_parser("methods", help="inspect registered similarity methods")
    methods_sub = methods.add_subparsers(dest="methods_command", required=True)
    methods_sub.add_parser("list", help="list available methods")
    datasets = subparsers.add_parser("datasets", help="inspect supported datasets")
    datasets_sub = datasets.add_subparsers(dest="datasets_command", required=True)
    datasets_sub.add_parser("list", help="list supported datasets")
    results = subparsers.add_parser("results")
    results_sub = results.add_subparsers(dest="results_command", required=True)
    rv = results_sub.add_parser("validate")
    rv.add_argument("run_dir")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "config":
        return (
            command_config_validate(args.file)
            if args.config_command == "validate"
            else command_config_resolve(args.file)
        )
    if args.command == "run":
        return command_run(
            args.config, resume=args.resume, dry_run=args.dry_run, force_stage=args.force_stage
        )
    if args.command == "analyze":
        return command_analyze(args.experiment, args.results_root, args.output)
    if args.command == "validate-data":
        return command_validate_data(args.path)
    if args.command == "perturb":
        return command_perturb(
            args.dataset, args.trajectory, args.transform, args.severity, args.seed, args.output
        )
    if args.command in {"methods", "datasets"}:
        return command_list(args.command)
    if args.command == "results":
        return command_results_validate(args.run_dir)
    if args.command == "dashboard":
        from trajsimbench.dashboard.app import main as dashboard_main

        result = dashboard_main(args.results_root)
        if result is not None:
            _json_print(result)
        return 0
    if args.command == "prepare":
        if args.dry_run:
            _json_print({"dataset": args.dataset, "config": args.config, "dry_run": True})
            return 0
        try:
            from trajsimbench.data.dataset import prepare_dataset
        except ImportError:
            message = (
                "dataset preparation is provided by the data scope; use synthetic Tiny run "
                "or install the data implementation"
            )
            print(
                message,
                file=sys.stderr,
            )
            return 2
        output = prepare_dataset(args.dataset, Path(args.config))
        _json_print({"dataset": args.dataset, "output": str(output)})
        return 0
    parser.error("unhandled command")
    return 2


# The project console entry point calls this function directly.  It uses
# argparse even when Typer is installed so the optional dependency does not
# change command behavior or import cost.
app = main


if __name__ == "__main__":
    raise SystemExit(main())
