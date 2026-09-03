"""Prepare the deterministic synthetic canonical fixture."""

from __future__ import annotations

import argparse

from trajsimbench.data.loaders.synthetic import prepare_synthetic


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/processed/synthetic/v1")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(prepare_synthetic(args.output, seed=args.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
