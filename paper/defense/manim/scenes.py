from __future__ import annotations

from typing import Iterable

import numpy as np
from manim import (
    Axes,
    Arrow,
    BarChart,
    BLUE_D,
    Circle,
    Create,
    CurvedArrow,
    DashedLine,
    DecimalNumber,
    Dot,
    DOWN,
    FadeIn,
    FadeOut,
    GREEN_C,
    GREY_B,
    LaggedStart,
    LEFT,
    Line,
    MathTex,
    Matrix,
    NumberLine,
    ORANGE,
    Rectangle,
    RED_C,
    RIGHT,
    RoundedRectangle,
    Scene,
    SurroundingRectangle,
    Text,
    TransformMatchingTex,
    UP,
    ValueTracker,
    VGroup,
    WHITE,
    Write,
    YELLOW_C,
    always_redraw,
)

P_MIN = 80.0
P_MAX = 160.0


def normal_pdf(x: float, mu: float, sigma: float) -> float:
    z = (x - mu) / sigma
    return float(np.exp(-0.5 * z * z) / (sigma * np.sqrt(2.0 * np.pi)))


def scene_title(text: str) -> Text:
    return Text(text, font_size=44, weight="BOLD", color=WHITE).to_edge(UP)


def card(
    label: str, color: str = BLUE_D, width: float = 3.3, height: float = 1.15
) -> VGroup:
    box = RoundedRectangle(corner_radius=0.15, width=width, height=height)
    box.set_stroke(color=color, width=2.0)
    box.set_fill(color=color, opacity=0.12)
    text = Text(label, font_size=24).move_to(box.get_center())
    return VGroup(box, text)


def to_matrix(values: Iterable[Iterable[float]], title: str, color: str) -> VGroup:
    mat = Matrix(
        [[f"{v:.2f}" for v in row] for row in values], h_buff=1.15, v_buff=0.75
    )
    header = Text(title, font_size=25, weight="BOLD", color=color).next_to(
        mat, UP, buff=0.2
    )
    frame = SurroundingRectangle(mat, color=color, buff=0.2)
    return VGroup(header, frame, mat)


class DefenseOpening(Scene):
    def construct(self) -> None:
        title = scene_title("PHANTOM Thesis Defense")
        subtitle = Text(
            "A mechanism-level defense for dynamic pricing under agentic traffic",
            font_size=27,
            color=GREY_B,
        ).next_to(title, DOWN, buff=0.35)

        roadmap = VGroup(
            Text("1) Define pricing power from first principles", font_size=30),
            Text("2) Show why agent saturation breaks it", font_size=30),
            Text(
                "3) Build a control loop from behavior to robust policy", font_size=30
            ),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.28)
        roadmap.next_to(subtitle, DOWN, buff=0.75).align_to(subtitle, LEFT)

        self.play(Write(title), FadeIn(subtitle, shift=UP * 0.2))
        self.play(
            LaggedStart(
                *[FadeIn(item, shift=RIGHT * 0.25) for item in roadmap], lag_ratio=0.18
            )
        )
        self.wait(0.9)


class COIFirstPrinciplesScene(Scene):
    def construct(self) -> None:
        title = scene_title("Cost of Information from First Principles")
        self.play(Write(title))

        setup = VGroup(
            MathTex(r"P\sim\pi(\tau)", font_size=44),
            MathTex(r"\underline p=\text{minimum viable price}", font_size=38),
            MathTex(r"M=P-\underline p", font_size=46, color=YELLOW_C),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22)
        setup.to_edge(LEFT).shift(UP * 0.55)

        self.play(
            LaggedStart(
                *[FadeIn(line, shift=RIGHT * 0.2) for line in setup], lag_ratio=0.2
            )
        )

        floor_x = 86.0
        mean_x = 116.0
        axes = (
            Axes(
                x_range=[80, 160, 10],
                y_range=[0.0, 0.04, 0.01],
                x_length=7.0,
                y_length=3.3,
                tips=False,
                axis_config={"stroke_width": 2},
            )
            .to_edge(RIGHT)
            .shift(DOWN * 0.2)
        )
        density = axes.plot(
            lambda x: normal_pdf(x, mean_x, 12.0),
            x_range=[80, 160],
            color=BLUE_D,
            stroke_width=6,
        )
        floor_line = Line(
            axes.c2p(floor_x, 0.0),
            axes.c2p(floor_x, 0.036),
            color=ORANGE,
            stroke_width=4,
        )
        mean_line = Line(
            axes.c2p(mean_x, 0.0),
            axes.c2p(mean_x, 0.036),
            color=GREEN_C,
            stroke_width=4,
        )
        floor_tag = (
            MathTex(r"\underline p", color=ORANGE)
            .scale(0.72)
            .next_to(floor_line, UP, buff=0.06)
        )
        mean_tag = (
            MathTex(r"\mathbb{E}[P]", color=GREEN_C)
            .scale(0.72)
            .next_to(mean_line, UP, buff=0.06)
        )
        coi_span = Line(
            axes.c2p(floor_x, 0.032),
            axes.c2p(mean_x, 0.032),
            color=YELLOW_C,
            stroke_width=6,
        )
        coi_tag = Text(
            "average information rent", font_size=18, color=YELLOW_C
        ).next_to(coi_span, UP, buff=0.05)

        chart = VGroup(
            axes,
            density,
            floor_line,
            mean_line,
            floor_tag,
            mean_tag,
            coi_span,
            coi_tag,
        )

        self.play(FadeIn(axes), FadeIn(density))
        self.play(
            FadeIn(floor_line), FadeIn(mean_line), FadeIn(floor_tag), FadeIn(mean_tag)
        )
        self.play(FadeIn(coi_span), FadeIn(coi_tag))
        self.play(
            FadeOut(setup, shift=LEFT * 0.15),
            chart.animate.scale(0.82).to_edge(RIGHT).shift(UP * 0.6),
        )

        eq1 = MathTex(r"\mathrm{COI}:=\mathbb{E}[M]", font_size=40)
        eq2 = MathTex(r"\mathrm{COI}=\mathbb{E}[P-\underline p]", font_size=40)
        eq3 = MathTex(
            r"\mathrm{COI}=\mathbb{E}[P]-\underline p", font_size=44, color=YELLOW_C
        )
        eq1.to_edge(LEFT).shift(UP * 0.45)
        eq2.move_to(eq1)
        eq3.move_to(eq1)

        self.play(Write(eq1))
        self.play(TransformMatchingTex(eq1, eq2))
        self.play(TransformMatchingTex(eq2, eq3))

        survival = MathTex(
            r"\mathrm{COI}=\int_{\underline p}^{\bar p}(1-F_\pi(p))\,dp",
            font_size=33,
            color=GREY_B,
        ).next_to(eq3, DOWN, aligned_edge=LEFT, buff=0.2)
        self.play(Write(survival))

        rationale = VGroup(
            Text("Why this definition is useful:", font_size=23, weight="BOLD"),
            Text("1) monetary meaning: premium over floor", font_size=20, color=GREY_B),
            Text("2) comparable across policies and runs", font_size=20, color=GREY_B),
            Text("3) maps directly to erosion analysis", font_size=20, color=GREY_B),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        rationale.next_to(survival, DOWN, aligned_edge=LEFT, buff=0.22).shift(UP * 0.1)
        self.play(FadeIn(rationale, shift=UP * 0.1))
        self.wait(1.0)


class COIOrderStatisticProofScene(Scene):
    def construct(self) -> None:
        title = scene_title("Why COI Erodes with Agent Saturation")
        self.play(Write(title))

        key = MathTex(r"p_{(1)}=\min(p_1,\ldots,p_N)", font_size=42, color=YELLOW_C)
        key.next_to(title, DOWN, buff=0.35)
        self.play(Write(key))

        number_line = NumberLine(
            x_range=[P_MIN, P_MAX, 10],
            length=10.8,
            include_numbers=True,
            decimal_number_config={"num_decimal_places": 0},
        ).shift(DOWN * 1.5)
        floor_marker = Line(
            number_line.n2p(P_MIN),
            number_line.n2p(P_MIN) + UP * 0.85,
            color=ORANGE,
            stroke_width=5,
        )
        floor_label = MathTex(r"\underline p", color=ORANGE).next_to(
            floor_marker, UP, buff=0.05
        )
        self.play(FadeIn(number_line), FadeIn(floor_marker), FadeIn(floor_label))

        rng = np.random.default_rng(17)
        current_group: VGroup | None = None
        current_info: VGroup | None = None

        for n in [1, 3, 8, 20]:
            draws = np.sort(rng.beta(2.4, 2.1, size=n) * (P_MAX - P_MIN) + P_MIN)
            dots = VGroup(
                *[
                    Dot(number_line.n2p(float(v)), radius=0.06, color=BLUE_D)
                    for v in draws
                ]
            )
            min_dot = Dot(number_line.n2p(float(draws[0])), radius=0.09, color=RED_C)
            min_tag = (
                MathTex(r"p_{(1)}", color=RED_C)
                .scale(0.65)
                .next_to(min_dot, UP, buff=0.08)
            )
            coi_n = Line(
                number_line.n2p(P_MIN) + UP * 0.68,
                number_line.n2p(float(draws[0])) + UP * 0.68,
                color=YELLOW_C,
                stroke_width=6,
            )
            step_group = VGroup(dots, min_dot, min_tag, coi_n)

            info = VGroup(
                Text(f"N = {n}", font_size=28),
                Text(f"min observed = {draws[0]:.2f}", font_size=24),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
            info.to_edge(LEFT).shift(UP * 0.55)
            info_box = VGroup(SurroundingRectangle(info, color=GREY_B, buff=0.18), info)

            if current_group is None:
                self.play(FadeIn(step_group), FadeIn(info_box))
            else:
                self.play(
                    FadeOut(current_group),
                    FadeOut(current_info),
                    FadeIn(step_group),
                    FadeIn(info_box),
                )
            current_group = step_group
            current_info = info_box
            self.wait(0.4)

        p1 = MathTex(
            r"\mathbb{P}(p_{(1)}>t)=\mathbb{P}(p_1>t,\ldots,p_N>t)", font_size=36
        )
        p2 = MathTex(r"\mathbb{P}(p_{(1)}>t)=[1-F(t)]^N", font_size=42, color=YELLOW_C)
        p1.to_edge(RIGHT).shift(UP * 0.55)
        p2.move_to(p1)

        self.play(Write(p1))
        self.play(TransformMatchingTex(p1, p2))

        tail_axes = (
            Axes(
                x_range=[0, 1, 0.2],
                y_range=[0, 1, 0.2],
                x_length=4.5,
                y_length=2.7,
                tips=False,
                axis_config={"stroke_width": 2},
            )
            .to_edge(RIGHT)
            .shift(DOWN * 0.85)
        )
        curve_1 = tail_axes.plot(
            lambda x: (1 - x) ** 1, x_range=[0, 1], color=BLUE_D, stroke_width=4
        )
        curve_4 = tail_axes.plot(
            lambda x: (1 - x) ** 4, x_range=[0, 1], color=GREEN_C, stroke_width=4
        )
        curve_16 = tail_axes.plot(
            lambda x: (1 - x) ** 16, x_range=[0, 1], color=RED_C, stroke_width=4
        )
        c_labels = VGroup(
            Text("N=1", font_size=18, color=BLUE_D),
            Text("N=4", font_size=18, color=GREEN_C),
            Text("N=16", font_size=18, color=RED_C),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        c_labels.next_to(tail_axes, RIGHT, buff=0.1)
        tail_x = MathTex(r"F(t)", font_size=24).next_to(tail_axes, DOWN, buff=0.05)
        tail_y = MathTex(r"[1-F(t)]^N", font_size=24).next_to(
            tail_axes, LEFT, buff=0.05
        )

        self.play(FadeIn(tail_axes), Create(curve_1), Create(curve_4), Create(curve_16))
        self.play(FadeIn(c_labels), FadeIn(tail_x), FadeIn(tail_y))

        e1 = MathTex(
            r"\mathbb{E}[p_{(1)}]=\underline p+\int_{\underline p}^{\bar p}[1-F(t)]^N\,dt",
            font_size=34,
        )
        e2 = MathTex(
            r"\lim_{N\to\infty}(\mathbb{E}[p_{(1)}]-\underline p)=0",
            font_size=42,
            color=YELLOW_C,
        )
        e1.to_edge(LEFT).shift(DOWN * 0.35)
        e2.next_to(e1, DOWN, aligned_edge=LEFT, buff=0.2)
        self.play(Write(e1), Write(e2))

        conclusion = Text(
            "As independent query count grows, realizable markup collapses.",
            font_size=24,
            color=GREY_B,
        )
        conclusion.to_edge(DOWN)
        self.play(FadeIn(conclusion, shift=UP * 0.1))
        self.wait(1.1)


class BehaviorKernelConstructionScene(Scene):
    def construct(self) -> None:
        title = scene_title("From Session Paths to Transition Kernels")
        self.play(Write(title))

        traj_h = Text(
            "human: start -> view -> detail -> cart -> purchase -> end",
            font_size=27,
            color=GREEN_C,
        )
        traj_a = Text(
            "agent: start -> view -> detail -> view -> detail -> end",
            font_size=27,
            color=RED_C,
        )
        trajectories = VGroup(traj_h, traj_a).arrange(
            DOWN, aligned_edge=LEFT, buff=0.18
        )
        trajectories.next_to(title, DOWN, buff=0.45).align_to(title, LEFT)
        self.play(
            LaggedStart(
                *[FadeIn(t, shift=RIGHT * 0.2) for t in trajectories], lag_ratio=0.25
            )
        )

        mle = MathTex(
            r"\hat P(s'\mid s)=\frac{N(s,s')}{\sum_k N(s,k)}",
            font_size=42,
            color=YELLOW_C,
        )
        mle.next_to(trajectories, DOWN, aligned_edge=LEFT, buff=0.35)
        self.play(Write(mle))

        counts = to_matrix(
            (
                (0.00, 8.00, 0.00, 0.00),
                (0.00, 2.00, 5.00, 1.00),
                (0.00, 3.00, 2.00, 4.00),
                (0.00, 1.00, 0.00, 6.00),
            ),
            "transition counts N(s,s')",
            color=BLUE_D,
        )
        probs = to_matrix(
            (
                (0.00, 1.00, 0.00, 0.00),
                (0.00, 0.25, 0.62, 0.13),
                (0.00, 0.33, 0.22, 0.45),
                (0.00, 0.14, 0.00, 0.86),
            ),
            "normalized kernel T",
            color=GREEN_C,
        )
        mats = (
            VGroup(counts, probs).arrange(RIGHT, buff=1.0).to_edge(DOWN).shift(UP * 0.2)
        )
        arrow = Arrow(counts.get_right(), probs.get_left(), buff=0.2, stroke_width=4)
        self.play(FadeIn(mats, shift=UP * 0.15), FadeIn(arrow))

        note = Text(
            "Kernel shape is the compact behavioral signature used downstream.",
            font_size=23,
            color=GREY_B,
        )
        note.next_to(mats, UP, buff=0.18)
        self.play(FadeIn(note, shift=UP * 0.1))
        self.wait(1.0)


class SeparabilitySignalScene(Scene):
    def construct(self) -> None:
        title = scene_title("Separability into a Control Signal")
        self.play(Write(title))

        human = to_matrix(
            (
                (0.05, 0.70, 0.20, 0.05),
                (0.05, 0.20, 0.60, 0.15),
                (0.10, 0.25, 0.30, 0.35),
                (0.00, 0.00, 0.00, 1.00),
            ),
            "human centroid T_H",
            color=GREEN_C,
        )
        agent = to_matrix(
            (
                (0.03, 0.82, 0.12, 0.03),
                (0.06, 0.55, 0.21, 0.18),
                (0.08, 0.48, 0.14, 0.30),
                (0.00, 0.00, 0.00, 1.00),
            ),
            "agent centroid T_A",
            color=RED_C,
        )
        kernels = VGroup(human, agent).arrange(RIGHT, buff=0.95).shift(UP * 0.45)
        self.play(FadeIn(kernels, shift=UP * 0.15))

        d_h = MathTex(r"\Delta_H=D_{KL}(\hat T'\parallel\bar T_H)", font_size=36)
        d_a = MathTex(r"\Delta_A=D_{KL}(\hat T'\parallel\bar T_A)", font_size=36)
        gap = MathTex(r"g=\Delta_H-\Delta_A", font_size=44, color=YELLOW_C)
        alpha = MathTex(r"\hat\alpha(\tau')=\sigma(\beta g)", font_size=40)
        eqs = VGroup(d_h, d_a, gap, alpha).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        eqs.next_to(kernels, DOWN, buff=0.32)
        self.play(LaggedStart(*[Write(eq) for eq in eqs], lag_ratio=0.18))

        self.play(
            FadeOut(kernels, shift=UP * 0.1), eqs.animate.to_edge(UP).shift(DOWN * 0.45)
        )

        mu_h, sigma_h = -3.35, 2.67
        mu_a, sigma_a = 1.65, 2.83
        axis = Axes(
            x_range=[-10, 10, 2],
            y_range=[0.0, 0.18, 0.03],
            x_length=10.3,
            y_length=3.6,
            tips=False,
            axis_config={"stroke_width": 2},
        ).next_to(eqs, DOWN, buff=0.45)
        x_tag = MathTex(r"g=\Delta_H-\Delta_A", font_size=30).next_to(
            axis, DOWN, buff=0.15
        )

        human_curve = axis.plot(
            lambda x: normal_pdf(x, mu_h, sigma_h),
            x_range=[-10, 10],
            color=BLUE_D,
            stroke_width=6,
        )
        agent_curve = axis.plot(
            lambda x: normal_pdf(x, mu_a, sigma_a),
            x_range=[-10, 10],
            color=RED_C,
            stroke_width=6,
        )
        h_label = Text("human", font_size=23, color=BLUE_D).next_to(
            axis.c2p(mu_h - 2.7, 0.09), LEFT, buff=0.12
        )
        a_label = Text("agent", font_size=23, color=RED_C).next_to(
            axis.c2p(mu_a + 2.5, 0.08), RIGHT, buff=0.12
        )

        boundary = DashedLine(
            axis.c2p(0.0, 0.0), axis.c2p(0.0, 0.165), color=GREY_B, stroke_width=2
        )
        boundary_tag = Text("decision boundary", font_size=17, color=GREY_B).next_to(
            boundary, UP, buff=0.08
        )

        g_obs = 1.6
        g_line = Line(
            axis.c2p(g_obs, 0.0), axis.c2p(g_obs, 0.145), color=YELLOW_C, stroke_width=4
        )
        g_dot = Dot(axis.c2p(g_obs, 0.145), color=YELLOW_C, radius=0.06)
        g_tag = (
            MathTex(r"g_{obs}", color=YELLOW_C)
            .scale(0.72)
            .next_to(g_dot, UP, buff=0.04)
        )

        self.play(FadeIn(axis), FadeIn(x_tag))
        self.play(Create(human_curve), Create(agent_curve))
        self.play(
            FadeIn(h_label), FadeIn(a_label), FadeIn(boundary), FadeIn(boundary_tag)
        )
        self.play(FadeIn(g_line), FadeIn(g_dot), FadeIn(g_tag))

        hint = Text(
            "Positive gap pushes the session score toward agent probability.",
            font_size=22,
            color=GREY_B,
        )
        hint.next_to(x_tag, DOWN, buff=0.1)
        self.play(FadeIn(hint, shift=UP * 0.1))
        self.wait(1.0)


class ContaminationGeneratorScene(Scene):
    def construct(self) -> None:
        title = scene_title("Contamination Generator G(alpha)")
        self.play(Write(title))

        human_pool = card("labeled human sessions", color=BLUE_D, width=4.1)
        agent_pool = card("synthetic agent sessions", color=RED_C, width=4.1)
        mixed_pool = card("mixed batch for training", color=YELLOW_C, width=4.4)

        top = (
            VGroup(human_pool, agent_pool)
            .arrange(RIGHT, buff=1.1)
            .next_to(title, DOWN, buff=0.55)
        )
        mixed_pool.next_to(top, DOWN, buff=1.25)

        a1 = Arrow(
            human_pool.get_bottom(),
            mixed_pool.get_top() + LEFT * 1.0,
            buff=0.1,
            stroke_width=4,
        )
        a2 = Arrow(
            agent_pool.get_bottom(),
            mixed_pool.get_top() + RIGHT * 1.0,
            buff=0.1,
            stroke_width=4,
        )

        self.play(FadeIn(top, shift=UP * 0.12), FadeIn(mixed_pool, shift=UP * 0.12))
        self.play(FadeIn(a1), FadeIn(a2))

        alpha_tracker = ValueTracker(0.15)
        bar_outline = Rectangle(
            width=6.1, height=0.42, stroke_color=WHITE, stroke_width=2
        ).next_to(mixed_pool, DOWN, buff=0.45)
        base_h = Rectangle(
            width=6.1, height=0.36, stroke_width=0, fill_color=BLUE_D, fill_opacity=0.35
        ).move_to(bar_outline)

        def make_agent_fill() -> Rectangle:
            width = max(0.02, 6.1 * alpha_tracker.get_value())
            rect = Rectangle(
                width=width,
                height=0.36,
                stroke_width=0,
                fill_color=RED_C,
                fill_opacity=0.68,
            )
            rect.move_to(bar_outline.get_right() + LEFT * (width / 2.0))
            return rect

        agent_fill = always_redraw(make_agent_fill)
        alpha_label = Text("alpha =", font_size=24).next_to(
            bar_outline, DOWN, buff=0.16
        )
        alpha_value = always_redraw(
            lambda: DecimalNumber(
                alpha_tracker.get_value(),
                num_decimal_places=2,
                font_size=28,
                color=YELLOW_C,
            ).next_to(alpha_label, RIGHT, buff=0.1)
        )
        left_tag = Text("human share", font_size=19, color=BLUE_D).next_to(
            bar_outline, LEFT, buff=0.15
        )
        right_tag = Text("agent share", font_size=19, color=RED_C).next_to(
            bar_outline, RIGHT, buff=0.15
        )

        self.play(FadeIn(bar_outline), FadeIn(base_h), FadeIn(agent_fill))
        self.play(
            FadeIn(alpha_label),
            FadeIn(alpha_value),
            FadeIn(left_tag),
            FadeIn(right_tag),
        )

        mix_eq = MathTex(
            r"Q(p)=(1-\alpha)\,\mathbb{E}_{\theta\sim D_H}[d(p;\theta)] + \alpha\,\mathbb{E}_{\theta\sim D_A}[d(p;\theta)]",
            font_size=30,
        ).next_to(bar_outline, DOWN, buff=0.45)
        interval = MathTex(
            r"\mathcal{A}_{\epsilon_\alpha}(\alpha_0)=\{\alpha:|\alpha-\alpha_0|\le\epsilon_\alpha\}",
            font_size=31,
            color=GREY_B,
        )
        interval.next_to(mix_eq, DOWN, buff=0.2)
        self.play(Write(mix_eq), Write(interval))

        self.play(alpha_tracker.animate.set_value(0.35), run_time=1.2)
        self.play(alpha_tracker.animate.set_value(0.60), run_time=1.2)
        self.play(alpha_tracker.animate.set_value(0.28), run_time=1.1)
        self.wait(0.9)


class RobustControlScene(Scene):
    def construct(self) -> None:
        title = scene_title("Distributionally Robust Control Layer")
        self.play(Write(title))

        objective = MathTex(
            r"\pi^*=\arg\max_\pi\min_{Q\in\mathcal U_\epsilon}\mathbb E_{d\sim Q}[R(p,d)-\lambda\,COI_{leak}(p,\tau') ]",
            font_size=32,
        ).next_to(title, DOWN, buff=0.4)
        reward = MathTex(
            r"r_t=R(p_t,\tilde q_t)-\lambda f(\tau_t')c_{info}",
            font_size=38,
            color=YELLOW_C,
        )
        reward.next_to(objective, DOWN, buff=0.25)
        self.play(Write(objective), Write(reward))

        plane = (
            Axes(
                x_range=[-3, 3, 1],
                y_range=[-3, 3, 1],
                x_length=5.6,
                y_length=5.6,
                tips=False,
                axis_config={"stroke_width": 1.8},
            )
            .to_edge(LEFT)
            .shift(DOWN * 0.45)
        )
        center = Dot(plane.c2p(0, 0), color=BLUE_D, radius=0.08)
        center_tag = (
            MathTex(r"\hat P_N", color=BLUE_D)
            .scale(0.75)
            .next_to(center, UP, buff=0.07)
        )
        ball = Circle(radius=1.75, color=YELLOW_C, stroke_width=3).move_to(center)
        ball_tag = (
            MathTex(r"\mathcal U_\epsilon", color=YELLOW_C)
            .scale(0.72)
            .next_to(ball, UP, buff=0.08)
        )

        q1 = Dot(plane.c2p(1.0, 0.7), color=GREEN_C)
        q2 = Dot(plane.c2p(-1.2, 0.9), color=RED_C)
        q3 = Dot(plane.c2p(0.3, -1.3), color=GREEN_C)
        q4 = Dot(plane.c2p(-0.9, -0.6), color=GREEN_C)
        q2_tag = Text("worst-case Q*", font_size=18, color=RED_C).next_to(
            q2, UP, buff=0.07
        )

        self.play(FadeIn(plane), FadeIn(center), FadeIn(center_tag))
        self.play(Create(ball), FadeIn(ball_tag))
        self.play(
            LaggedStart(*[FadeIn(dot) for dot in [q1, q2, q3, q4]], lag_ratio=0.14)
        )
        self.play(FadeIn(q2_tag, shift=UP * 0.08))

        chooser = Arrow(
            q2.get_right() + RIGHT * 0.15,
            q2.get_right() + RIGHT * 0.95,
            buff=0.05,
            color=RED_C,
            stroke_width=4,
        )
        policy_card = (
            card("policy update", color=RED_C, width=2.8, height=0.85)
            .to_edge(RIGHT)
            .shift(DOWN * 0.6)
        )
        self.play(FadeIn(chooser), FadeIn(policy_card, shift=LEFT * 0.15))

        note = Text(
            "Train against plausible demand shifts, not just one estimate.",
            font_size=22,
            color=GREY_B,
        )
        note.to_edge(DOWN)
        self.play(FadeIn(note, shift=UP * 0.1))
        self.wait(1.0)


class SystemLoopScene(Scene):
    def construct(self) -> None:
        title = scene_title("Online + Offline Defense Loop")
        self.play(Write(title))

        web = card("Web App", color=BLUE_D)
        kafka = card("Kafka Streams", color=YELLOW_C)
        kernels = card("Kernel + KL estimator", color=GREEN_C, width=4.0)
        generator = card("Generator G(alpha)", color=GREEN_C)
        policy = card("DR-RL policy", color=ORANGE)
        provider = card("Pricing provider", color=BLUE_D)

        top = VGroup(web, kafka, kernels).arrange(RIGHT, buff=0.55).shift(UP * 0.95)
        bottom = (
            VGroup(generator, policy, provider)
            .arrange(RIGHT, buff=0.7)
            .next_to(top, DOWN, buff=1.15)
        )
        arrows = VGroup(
            Arrow(web.get_right(), kafka.get_left(), buff=0.12, stroke_width=4),
            Arrow(kafka.get_right(), kernels.get_left(), buff=0.12, stroke_width=4),
            Arrow(kernels.get_bottom(), generator.get_top(), buff=0.12, stroke_width=4),
            Arrow(generator.get_right(), policy.get_left(), buff=0.12, stroke_width=4),
            Arrow(policy.get_right(), provider.get_left(), buff=0.12, stroke_width=4),
            CurvedArrow(
                provider.get_top(), web.get_bottom(), angle=1.3, stroke_width=4
            ),
        )

        self.play(
            LaggedStart(
                *[FadeIn(node, shift=UP * 0.1) for node in VGroup(top, bottom)],
                lag_ratio=0.14,
            )
        )
        self.play(LaggedStart(*[FadeIn(a) for a in arrows], lag_ratio=0.08))

        labels = VGroup(
            Text("behavior events + price queries", font_size=19).next_to(
                arrows[1], UP, buff=0.08
            ),
            Text("inner worst-case step", font_size=19).next_to(
                arrows[3], DOWN, buff=0.12
            ),
            Text("serve updated prices", font_size=19).next_to(
                arrows[4], UP, buff=0.08
            ),
        )
        self.play(LaggedStart(*[FadeIn(l) for l in labels], lag_ratio=0.2))
        self.wait(1.0)


class ObjectiveAndResultsScene(Scene):
    def construct(self) -> None:
        title = scene_title("Early Experimental Signal")
        self.play(Write(title))

        objective_chart = BarChart(
            values=[3.41, 3.91],
            bar_names=["robust", "non-robust"],
            y_range=[0, 5, 1],
            y_length=2.9,
            x_length=4.8,
            bar_colors=[GREEN_C, RED_C],
        )
        objective_label = Text("objective (x1e5)", font_size=21).next_to(
            objective_chart, UP, buff=0.1
        )

        revenue_chart = BarChart(
            values=[3.80, 4.18],
            bar_names=["robust", "non-robust"],
            y_range=[0, 5, 1],
            y_length=2.9,
            x_length=4.8,
            bar_colors=[GREEN_C, RED_C],
        )
        revenue_label = Text("revenue (x1e5)", font_size=21).next_to(
            revenue_chart, UP, buff=0.1
        )

        charts = VGroup(
            VGroup(objective_label, objective_chart),
            VGroup(revenue_label, revenue_chart),
        ).arrange(RIGHT, buff=0.85)
        charts.next_to(title, DOWN, buff=0.7)
        self.play(FadeIn(charts, shift=UP * 0.2))

        pairwise = VGroup(
            Text("pairwise win counts", font_size=24, weight="BOLD"),
            Text("objective: robust beats baseline in 13 / 40", font_size=22),
            Text("revenue: robust beats baseline in 16 / 40", font_size=22),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.13)
        pairwise.next_to(charts, DOWN, buff=0.35)
        self.play(
            LaggedStart(
                *[FadeIn(row, shift=RIGHT * 0.15) for row in pairwise], lag_ratio=0.18
            )
        )

        caution = Text(
            "Interpretation: defense effect is real but regime-dependent and needs calibration.",
            font_size=22,
            color=GREY_B,
        ).to_edge(DOWN)
        self.play(FadeIn(caution, shift=UP * 0.1))
        self.wait(1.1)


class TakeawayScene(Scene):
    def construct(self) -> None:
        title = scene_title("Takeaways")
        self.play(Write(title))

        bullets = VGroup(
            Text("COI gives a clean monetary KPI for pricing power.", font_size=32),
            Text(
                "Behavioral KL separability becomes a live control signal.",
                font_size=32,
            ),
            Text(
                "DR-RL with ambiguity sets protects against contamination shift.",
                font_size=32,
            ),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.32)
        bullets.next_to(title, DOWN, buff=0.7).align_to(title, LEFT)
        self.play(
            LaggedStart(
                *[FadeIn(item, shift=RIGHT * 0.2) for item in bullets], lag_ratio=0.2
            )
        )

        final = Text(
            "From mechanism failure to implementable defense loop.",
            font_size=29,
            color=YELLOW_C,
        )
        final.to_edge(DOWN)
        self.play(FadeIn(final, shift=UP * 0.1))
        self.wait(1.0)


SCENE_ORDER = [
    "DefenseOpening",
    "COIFirstPrinciplesScene",
    "COIOrderStatisticProofScene",
    "BehaviorKernelConstructionScene",
    "SeparabilitySignalScene",
    "ContaminationGeneratorScene",
    "RobustControlScene",
    "SystemLoopScene",
    "ObjectiveAndResultsScene",
    "TakeawayScene",
]
