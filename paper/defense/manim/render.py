from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from scenes import SCENE_ORDER


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render thesis-defense Manim scenes")
    parser.add_argument(
        "--quality",
        default="qm",
        choices=["ql", "qm", "qh", "qk"],
        help="Manim quality preset",
    )
    parser.add_argument(
        "--scene",
        action="append",
        dest="scenes",
        help="Scene name; repeat flag to render many",
    )
    parser.add_argument(
        "--preview", action="store_true", help="Open video after each render"
    )
    parser.add_argument(
        "--list", action="store_true", help="List available scenes and exit"
    )
    return parser.parse_args()


def validate_requested(requested: list[str]) -> list[str]:
    missing = [name for name in requested if name not in SCENE_ORDER]
    if missing:
        choices = ", ".join(SCENE_ORDER)
        raise ValueError(f"Unknown scenes: {', '.join(missing)}. Choices: {choices}")
    return requested


def run_manim(scene_file: Path, scene_name: str, quality: str, preview: bool) -> None:
    cmd = [sys.executable, "-m", "manim"]
    if preview:
        cmd.append("-p")
    cmd.extend([f"-{quality}", str(scene_file), scene_name])
    subprocess.run(cmd, cwd=scene_file.parent, check=True)


def main() -> int:
    args = parse_args()
    if args.list:
        for scene in SCENE_ORDER:
            print(scene)
        return 0

    scenes = validate_requested(args.scenes) if args.scenes else list(SCENE_ORDER)
    scene_file = Path(__file__).resolve().parent / "scenes.py"

    try:
        for scene_name in scenes:
            run_manim(
                scene_file=scene_file,
                scene_name=scene_name,
                quality=args.quality,
                preview=args.preview,
            )
    except FileNotFoundError:
        print(
            "manim executable not found. Install Manim in your Python environment.",
            file=sys.stderr,
        )
        return 2
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as exc:
        return exc.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
