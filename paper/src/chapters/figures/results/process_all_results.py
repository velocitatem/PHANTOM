from __future__ import annotations

import argparse
from pathlib import Path

from process_first_sweep import run as run_first_sweep
from process_ppo_benchmark import run as run_ppo_benchmark


def _default_output_dir() -> Path:
    return Path(__file__).resolve().parent / "generated" / "legacy"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process all result CSV exports for paper figures"
    )
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    parser.add_argument("--include-non-finished", action="store_true")
    parser.add_argument("--top-n", type=int, default=25)
    args = parser.parse_args()

    written: list[Path] = []
    written.extend(
        run_ppo_benchmark(
            input_path=Path(__file__).resolve().parents[5]
            / "tpu_orchestration"
            / "results"
            / "ppo_benchmark.csv",
            output_dir=args.output_dir,
            include_non_finished=bool(args.include_non_finished),
        )
    )
    written.extend(
        run_first_sweep(
            input_path=Path(__file__).resolve().parents[5]
            / "tpu_orchestration"
            / "results"
            / "first_sweep.csv",
            output_dir=args.output_dir,
            include_non_finished=bool(args.include_non_finished),
            top_n=int(args.top_n),
        )
    )

    for path in written:
        print(path)


if __name__ == "__main__":
    main()
