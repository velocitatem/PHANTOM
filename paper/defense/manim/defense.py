"""Manim entry module only.

Scene implementations are in scenes/main.py and scenes/appendix.py. Manim names
output folders after the file you pass to the CLI; pointing everything at this
file keeps all MP4s under media/videos/defense/ instead of splitting by source file.
"""

from __future__ import annotations

import importlib

from manim import Scene

_modname = __name__

for _mod in ("scenes.main", "scenes.appendix"):
    m = importlib.import_module(_mod)
    for _name, _val in list(vars(m).items()):
        if _name.startswith("_"):
            continue
        if isinstance(_val, type) and issubclass(_val, Scene) and _val is not Scene:
            _val.__module__ = _modname
            globals()[_name] = _val
