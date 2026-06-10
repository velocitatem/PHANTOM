from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from scenes.appendix import BEHAVIOR_SCENES, COI_SCENES
from scenes.main import POSTER_SCENES, SCENE_ORDER as MAIN_SCENES

# ---------------------------------------------------------------------------
# Batch render: groups are just ordered lists of scene class names.
# Every scene is rendered via defense.py so outputs stay in media/videos/defense/.
# Scene code itself lives in scenes/main.py and scenes/appendix.py.
# ---------------------------------------------------------------------------


def _ordered_unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    return [item for item in items if not (item in seen or seen.add(item))]


FINAL_CORE = [
    "DefenseOpening",
    "CardMarketAnalogyScene",
    "COIFirstPrinciplesScene",
    "COIOrderStatisticProofScene",
    "BehaviorKernelConstructionScene",
    "SeparabilitySignalScene",
    "ContaminationGeneratorScene",
    "RewardAndLeakageScene",
    "StackelbergAmbiguityScene",
    "RobustControlScene",
    "SystemLoopScene",
    "ObjectiveAndResultsScene",
]

SCENE_GROUPS: dict[str, list[str]] = {
    "poster": list(POSTER_SCENES),
    "final-core": FINAL_CORE,
    "final-full": list(MAIN_SCENES),
    "behavior-appendix": list(BEHAVIOR_SCENES),
    "coi-appendix": list(COI_SCENES),
}

SCENE_GROUPS["all"] = _ordered_unique(
    [
        *SCENE_GROUPS["final-full"],
        *SCENE_GROUPS["poster"],
        *SCENE_GROUPS["behavior-appendix"],
        *SCENE_GROUPS["coi-appendix"],
    ]
)

ENTRY = "defense.py"
SCENE_TO_FILE: dict[str, str] = {name: ENTRY for name in SCENE_GROUPS["all"]}

DEFAULT_GROUP = "final-core"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-render scenes. Code: scenes/main.py + scenes/appendix.py. "
            "Manim entry: defense.py. Output: media/videos/defense/<quality>/"
        )
    )
    parser.add_argument(
        "--quality",
        default="qm",
        choices=["ql", "qm", "qh", "qk"],
        help="Manim quality preset",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--scene",
        action="append",
        dest="scenes",
        help="Scene name; repeat to render many",
    )
    selection.add_argument(
        "--group",
        choices=sorted(SCENE_GROUPS.keys()),
        default=DEFAULT_GROUP,
        help=f"Named list of scenes (default: {DEFAULT_GROUP})",
    )
    selection.add_argument("--all", action="store_true", help="Render every scene")
    parser.add_argument(
        "--media-dir",
        default="media",
        help="Relative to this folder (default: media)",
    )
    parser.add_argument("--preview", action="store_true", help="Open each video")
    parser.add_argument("--list", action="store_true", help="Print groups and exit")
    return parser.parse_args()


def validate_requested(requested: list[str]) -> list[str]:
    missing = [name for name in requested if name not in SCENE_TO_FILE]
    if missing:
        choices = ", ".join(SCENE_TO_FILE.keys())
        raise ValueError(
            f"Unknown scenes: {', '.join(missing)}\nAvailable choices: {choices}"
        )
    return requested


def resolve_scenes(args: argparse.Namespace) -> list[str]:
    if args.all:
        return list(SCENE_GROUPS["all"])
    if args.scenes:
        return validate_requested(args.scenes)
    return list(SCENE_GROUPS[args.group])


def run_manim(
    scene_file: Path,
    scene_name: str,
    quality: str,
    preview: bool,
    working_dir: Path,
    media_dir: str,
    pythonpath: str,
) -> None:
    env = os.environ.copy()
    prev = env.get("PYTHONPATH")
    env["PYTHONPATH"] = pythonpath if not prev else f"{pythonpath}:{prev}"

    cmd = [sys.executable, "-m", "manim"]
    if preview:
        cmd.append("-p")
    cmd.extend(["--media_dir", media_dir])
    cmd.extend([f"-{quality}", str(scene_file), scene_name])
    subprocess.run(cmd, cwd=working_dir, check=True, env=env)


def main() -> int:
    args = parse_args()
    if args.list:
        for group_name in sorted(SCENE_GROUPS):
            print(f"[{group_name}]")
            for scene in SCENE_GROUPS[group_name]:
                print(f"  {scene}")
        return 0

    root = Path(__file__).resolve().parent
    py_path = str(root)
    names = resolve_scenes(args)

    try:
        for scene_name in names:
            scene_file = root / SCENE_TO_FILE[scene_name]
            run_manim(
                scene_file=scene_file,
                scene_name=scene_name,
                quality=args.quality,
                preview=args.preview,
                working_dir=root,
                media_dir=args.media_dir,
                pythonpath=py_path,
            )
    except FileNotFoundError:
        print("manim not found.", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as exc:
        return exc.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
