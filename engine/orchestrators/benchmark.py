from __future__ import annotations


def run_benchmark_cli(raw_args: list[str] | None = None) -> None:
    from ..benchmark import run_cli

    run_cli(raw_args)
