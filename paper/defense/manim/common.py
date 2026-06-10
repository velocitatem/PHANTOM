from __future__ import annotations

from typing import Iterable

import numpy as np
from manim import (
    Arrow,
    BLUE_D,
    CurvedArrow,
    DOWN,
    DashedLine,
    GREEN_C,
    GREY_B,
    LEFT,
    Line,
    MathTex,
    Matrix,
    RIGHT,
    RoundedRectangle,
    SurroundingRectangle,
    Text,
    UP,
    VGroup,
    config,
)

P_MIN = 80.0
P_MAX = 160.0
LIGHT_BG = "#F8F8F4"
INK = "#1E1E1E"
AXIS_INK = "#2C2C2C"
HIGHLIGHT = "#8F5F00"

config.background_color = LIGHT_BG
Text.set_default(color=INK)
MathTex.set_default(color=INK)
Line.set_default(color=AXIS_INK)
Arrow.set_default(color=AXIS_INK)
CurvedArrow.set_default(color=AXIS_INK)
DashedLine.set_default(color=AXIS_INK)


def normal_pdf(x: float, mu: float, sigma: float) -> float:
    z = (x - mu) / sigma
    return float(np.exp(-0.5 * z * z) / (sigma * np.sqrt(2.0 * np.pi)))


def scene_title(text: str) -> Text:
    return Text(text, font_size=44, weight="BOLD", color=INK).to_edge(UP)


def card(
    label: str,
    color: str = BLUE_D,
    width: float = 3.3,
    height: float = 1.15,
    font_size: float = 24,
) -> VGroup:
    box = RoundedRectangle(corner_radius=0.15, width=width, height=height)
    box.set_stroke(color=color, width=2.0)
    box.set_fill(color=color, opacity=0.12)
    text = Text(label, font_size=font_size).move_to(box.get_center())
    return VGroup(box, text)


def to_matrix(
    values: Iterable[Iterable[float]],
    title: str,
    color: str,
    header_buff: float = 0.28,
    fmt: str = ".2f",
) -> VGroup:
    mat = Matrix(
        [[f"{v:{fmt}}" for v in row] for row in values], h_buff=1.15, v_buff=0.75
    )
    header = Text(title, font_size=25, weight="BOLD", color=color).next_to(
        mat, UP, buff=header_buff
    )
    frame = SurroundingRectangle(mat, color=color, buff=0.2)
    return VGroup(header, frame, mat)


def rank_from_scale(scale: int) -> str:
    clamped = max(1, min(scale, 10))
    return "A" if clamped == 1 else str(clamped)


def actor_face_card(
    rank: str,
    role: str,
    accent: str,
    width: float = 1.6,
    height: float = 2.25,
    show_role: bool = True,
) -> VGroup:
    frame = RoundedRectangle(corner_radius=0.1, width=width, height=height)
    frame.set_stroke(color=AXIS_INK, width=2.0)
    frame.set_fill(color="#FFFFFF", opacity=1.0)

    top_rank = Text(rank, font_size=30, color=accent).move_to(
        frame.get_corner(UP + LEFT) + RIGHT * 0.2 + DOWN * 0.22
    )
    bottom_rank = (
        Text(rank, font_size=30, color=accent)
        .rotate(np.pi)
        .move_to(frame.get_corner(DOWN + RIGHT) + LEFT * 0.2 + UP * 0.22)
    )
    center_rank = Text(rank, font_size=56, weight="BOLD", color=accent).move_to(
        frame.get_center() + UP * 0.03
    )

    parts = [frame, top_rank, bottom_rank, center_rank]
    if show_role:
        role_label = Text(role, font_size=18, color=GREY_B).next_to(
            frame, DOWN, buff=0.08
        )
        parts.append(role_label)
    return VGroup(*parts)


def product_suit_card(
    suit: str,
    scale: int,
    accent: str,
    width: float = 1.86,
    height: float = 1.04,
    show_label: bool = False,
) -> tuple[VGroup, Text]:
    frame = RoundedRectangle(corner_radius=0.08, width=width, height=height)
    frame.set_stroke(color=AXIS_INK, width=2.0)
    frame.set_fill(color="#FFFFFF", opacity=1.0)

    suit_left = Text(suit, font_size=28, color=accent).move_to(
        frame.get_left() + RIGHT * 0.22
    )
    suit_right = Text(suit, font_size=28, color=accent).move_to(
        frame.get_right() + LEFT * 0.22
    )
    scale_text = Text(
        rank_from_scale(scale),
        font_size=40,
        weight="BOLD",
        color=accent,
    ).move_to(frame.get_center())

    parts = [frame, suit_left, suit_right, scale_text]
    if show_label:
        scale_label = Text("scale", font_size=14, color=GREY_B).next_to(
            frame, DOWN, buff=0.04
        )
        parts.append(scale_label)
    return VGroup(*parts), scale_text


def private_valuation_card(value: int, show_label: bool = False) -> VGroup:
    frame = RoundedRectangle(corner_radius=0.08, width=1.86, height=1.04)
    frame.set_stroke(color=AXIS_INK, width=2.0)
    frame.set_fill(color="#FFFFFF", opacity=1.0)

    rank = Text(
        rank_from_scale(value), font_size=40, weight="BOLD", color=GREEN_C
    ).move_to(frame.get_center())
    left_tag = Text("v", font_size=28, color=INK).move_to(
        frame.get_left() + RIGHT * 0.22
    )
    right_tag = Text("*", font_size=28, color=INK).move_to(
        frame.get_right() + LEFT * 0.22
    )

    parts = [frame, left_tag, right_tag, rank]
    if show_label:
        title = Text("private value", font_size=14, color=GREY_B).next_to(
            frame, DOWN, buff=0.04
        )
        parts.append(title)
    return VGroup(*parts)
