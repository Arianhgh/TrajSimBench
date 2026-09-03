"""Run the reproducible CPU-only Tiny benchmark."""

from __future__ import annotations

import argparse

from trajsimbench.orchestration.runner import run_experiment


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ci/tiny_synthetic.yaml")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run_experiment(args.config, resume=args.resume, output_root=args.output_root)
    print(result.run_dir if hasattr(result, "run_dir") else result)
    return 0 if getattr(result, "status", "failed") == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
