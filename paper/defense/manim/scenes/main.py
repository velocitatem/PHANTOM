from __future__ import annotations

import numpy as np
from manim import (
    Axes,
    Arrow,
    BarChart,
    BLUE_D,
    Circle,
    Circumscribe,
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
    Indicate,
    LaggedStart,
    LEFT,
    Line,
    MathTex,
    NumberLine,
    ORANGE,
    Rectangle,
    RED_C,
    RIGHT,
    Scene,
    SurroundingRectangle,
    Text,
    Transform,
    UP,
    ValueTracker,
    VGroup,
    Write,
    always_redraw,
    smooth,
)
from common import (
    AXIS_INK,
    HIGHLIGHT,
    INK,
    P_MAX,
    P_MIN,
    actor_face_card,
    card,
    normal_pdf,
    private_valuation_card,
    product_suit_card,
    rank_from_scale,
    scene_title,
    to_matrix,
)


class DefenseOpening(Scene):
    def construct(self) -> None:
        title = scene_title("PHANTOM")
        tag = MathTex(
            r"\text{dynamic pricing under agent-mediated traffic}",
            font_size=30,
            color=GREY_B,
        ).next_to(title, DOWN, buff=0.2)
        self.play(Write(title), FadeIn(tag, shift=UP * 0.12))

        dist_axes = Axes(
            x_range=[-6, 6, 2],
            y_range=[0.0, 0.2, 0.05],
            x_length=3.1,
            y_length=1.75,
            tips=False,
            axis_config={"stroke_width": 1.8, "color": AXIS_INK},
        )
        dist_h = dist_axes.plot(
            lambda x: normal_pdf(x, -1.9, 1.6),
            x_range=[-6, 6],
            color=BLUE_D,
            stroke_width=4,
        )
        dist_a = dist_axes.plot(
            lambda x: normal_pdf(x, 1.8, 1.8),
            x_range=[-6, 6],
            color=RED_C,
            stroke_width=4,
        )
        dist_block = VGroup(
            dist_axes,
            dist_h,
            dist_a,
            MathTex(r"g=\Delta_H-\Delta_A", font_size=22, color=GREY_B).next_to(
                dist_axes, DOWN, buff=0.05
            ),
        )

        tail_axes = Axes(
            x_range=[0, 1, 0.2],
            y_range=[0, 1, 0.2],
            x_length=3.1,
            y_length=1.75,
            tips=False,
            axis_config={"stroke_width": 1.8, "color": AXIS_INK},
        )
        tail_n1 = tail_axes.plot(
            lambda x: (1 - x) ** 1,
            x_range=[0, 1],
            color=GREEN_C,
            stroke_width=4,
        )
        tail_n8 = tail_axes.plot(
            lambda x: (1 - x) ** 8,
            x_range=[0, 1],
            color=HIGHLIGHT,
            stroke_width=4,
        )
        tail_block = VGroup(
            tail_axes,
            tail_n1,
            tail_n8,
            MathTex(r"[1{-}F(t)]^N", font_size=22, color=GREY_B).next_to(
                tail_axes, DOWN, buff=0.05
            ),
        )

        control_eq = MathTex(
            r"\hat\alpha(\tau')\Rightarrow\pi^*",
            font_size=36,
            color=HIGHLIGHT,
        )
        control_box = SurroundingRectangle(control_eq, color=HIGHLIGHT, buff=0.12)
        control_block = VGroup(control_box, control_eq)

        preview = VGroup(dist_block, tail_block, control_block).arrange(
            RIGHT, buff=0.5
        )
        preview.next_to(tag, DOWN, buff=0.48)
        preview_caption = MathTex(
            r"\text{separability}\ \to\ \text{tail}\ \to\ \text{policy}",
            font_size=24,
            color=GREY_B,
        ).next_to(preview, UP, buff=0.1)

        f_arrow_1 = Arrow(dist_block.get_right(), tail_block.get_left(), buff=0.08)
        f_arrow_2 = Arrow(tail_block.get_right(), control_block.get_left(), buff=0.08)

        self.play(FadeIn(preview_caption, shift=UP * 0.1))
        self.play(FadeIn(dist_block), FadeIn(tail_block), FadeIn(control_block))
        self.play(FadeIn(f_arrow_1), FadeIn(f_arrow_2))
        self.play(Indicate(control_block, color=HIGHLIGHT, run_time=1.0))
        self.wait(0.75)


class CardMarketAnalogyScene(Scene):
    def construct(self) -> None:
        title = scene_title("Card Analogy: Platform, Customer, Agent")
        self.play(Write(title))

        subtitle = MathTex(
            r"K\text{ (platform)},\ Q\text{ (customer)},\ J\text{ (agent)},\ \clubsuit\text{--}\diamondsuit=\text{SKUs}",
            font_size=22,
            color=GREY_B,
        ).next_to(title, DOWN, buff=0.14)
        self.play(FadeIn(subtitle, shift=UP * 0.05))

        king = actor_face_card(
            rank="K", role="platform", accent=ORANGE, show_role=False
        )
        king.move_to(LEFT * 5.35 + DOWN * 0.35)

        queen_home = RIGHT * 3.2 + DOWN * 0.28
        queen = actor_face_card(
            rank="Q", role="customer", accent=BLUE_D, show_role=False
        )
        queen.move_to(queen_home)

        valuation = private_valuation_card(value=5).next_to(queen, RIGHT, buff=0.35)

        specs = [
            ("C", INK, 4),
            ("H", RED_C, 6),
            ("S", INK, 5),
            ("D", RED_C, 3),
        ]
        scales = [initial for _, _, initial in specs]
        products = VGroup()
        scale_tokens: list[Text] = []
        for suit, color, initial in specs:
            product_card, token = product_suit_card(
                suit=suit, scale=initial, accent=color
            )
            products.add(product_card)
            scale_tokens.append(token)

        products.arrange(DOWN, buff=0.15).move_to(LEFT * 1.75 + DOWN * 0.55)

        actor_link = Arrow(
            king.get_right(),
            products.get_left(),
            buff=0.15,
            color=HIGHLIGHT,
            stroke_width=3.6,
        )

        self.play(
            FadeIn(king, shift=RIGHT * 0.2),
            FadeIn(products, shift=UP * 0.15),
            FadeIn(queen, shift=LEFT * 0.2),
            FadeIn(valuation, shift=LEFT * 0.2),
        )
        self.play(FadeIn(actor_link))

        stage = Text(
            "Stage 1: queen browses directly and visited products rise in scale.",
            font_size=21,
            color=GREY_B,
        ).to_edge(DOWN)
        self.play(FadeIn(stage, shift=UP * 0.08))

        direct_visits = [1, 2]
        for idx in direct_visits:
            target = products[idx]
            demand_box = SurroundingRectangle(target, color=BLUE_D, buff=0.06)
            king_box = SurroundingRectangle(king[0], color=HIGHLIGHT, buff=0.07)

            self.play(
                queen.animate.move_to(target.get_right() + RIGHT * 0.9),
                run_time=0.7,
            )
            self.play(Create(demand_box), run_time=0.2)

            scales[idx] = min(10, scales[idx] + 2)
            new_scale = Text(
                rank_from_scale(scales[idx]),
                font_size=40,
                weight="BOLD",
                color=specs[idx][1],
            ).move_to(scale_tokens[idx])
            self.play(
                Create(king_box),
                Transform(scale_tokens[idx], new_scale),
                run_time=0.5,
            )
            self.play(FadeOut(king_box), FadeOut(demand_box), run_time=0.18)

        self.play(queen.animate.move_to(queen_home), run_time=0.7)

        stage_two = Text(
            "Stage 2: queen hires jack to search every card before deciding.",
            font_size=21,
            color=GREY_B,
        ).to_edge(DOWN)
        self.play(Transform(stage, stage_two))

        jack = actor_face_card(
            rank="J", role="agent", accent=RED_C, show_role=False
        ).scale(0.95)
        jack.next_to(queen, LEFT, buff=0.35)
        hire_arrow = Arrow(
            queen.get_left(),
            jack.get_right(),
            buff=0.08,
            color=HIGHLIGHT,
            stroke_width=2.6,
        )
        self.play(FadeIn(jack, shift=RIGHT * 0.16), FadeIn(hire_arrow))
        self.play(FadeOut(hire_arrow), run_time=0.2)

        for idx, target in enumerate(products):
            demand_box = SurroundingRectangle(target, color=RED_C, buff=0.05)
            king_box = SurroundingRectangle(king[0], color=HIGHLIGHT, buff=0.07)

            self.play(
                jack.animate.move_to(target.get_right() + RIGHT * 0.62),
                run_time=0.32,
            )
            self.play(Create(demand_box), run_time=0.17)

            scales[idx] = min(10, scales[idx] + 1)
            new_scale = Text(
                rank_from_scale(scales[idx]),
                font_size=40,
                weight="BOLD",
                color=specs[idx][1],
            ).move_to(scale_tokens[idx])
            self.play(
                Create(king_box), Transform(scale_tokens[idx], new_scale), run_time=0.38
            )
            self.play(
                FadeOut(king_box),
                FadeOut(demand_box),
                run_time=0.15,
            )

        self.play(jack.animate.next_to(queen, LEFT, buff=0.35), run_time=0.55)

        report_arrow = Arrow(
            jack.get_right(),
            queen.get_left(),
            buff=0.08,
            color=GREEN_C,
            stroke_width=2.6,
        )
        self.play(FadeIn(report_arrow))

        best_idx = int(np.argmin(scales))
        best_card = products[best_idx]
        choice_box = SurroundingRectangle(best_card, color=GREEN_C, buff=0.07)
        stage_three = Text(
            "Decision rule: buy when private value v* exceeds shown scale.",
            font_size=21,
            color=GREY_B,
        ).to_edge(DOWN)

        self.play(
            Transform(stage, stage_three),
            queen.animate.move_to(best_card.get_right() + RIGHT * 0.9),
            Create(choice_box),
            run_time=0.95,
        )
        self.play(
            FadeOut(jack),
            FadeOut(report_arrow),
            FadeOut(actor_link),
            FadeOut(subtitle),
        )
        self.wait(1.0)


class COIFirstPrinciplesScene(Scene):
    def construct(self) -> None:
        title = scene_title("Cost of Information from First Principles")
        self.play(Write(title))

        setup = VGroup(
            MathTex(r"P\sim\pi(\tau)", font_size=44),
            MathTex(r"\underline p=\text{reservation price}", font_size=38),
            MathTex(r"M=P-\underline p", font_size=46, color=HIGHLIGHT),
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
                axis_config={"stroke_width": 2, "color": AXIS_INK},
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
            color=HIGHLIGHT,
            stroke_width=6,
        )
        coi_tag = MathTex(
            r"\mathbb{E}[P]-\underline p", font_size=22, color=HIGHLIGHT
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

        coi_left = MathTex(r"\mathrm{COI}=\mathbb{E}[", font_size=42)
        coi_mid = MathTex(r"M", font_size=42)
        coi_right = MathTex(r"]", font_size=42)
        coi_eq = VGroup(coi_left, coi_mid, coi_right).arrange(RIGHT, buff=0.04)
        coi_eq.to_edge(LEFT).shift(UP * 0.45)

        self.play(Write(coi_left), FadeIn(coi_mid, shift=UP * 0.05), Write(coi_right))

        expanded_mid = MathTex(r"P-\underline p", font_size=42)
        expanded_mid.move_to(coi_mid, aligned_edge=LEFT)
        self.play(
            Transform(coi_mid, expanded_mid),
            coi_right.animate.next_to(coi_mid, RIGHT, buff=0.04),
        )
        self.play(coi_eq.animate.set_color(HIGHLIGHT))
        self.play(Indicate(coi_eq, color=HIGHLIGHT, run_time=0.85))

        survival = MathTex(
            r"\mathrm{COI}=\int_{\underline p}^{\bar p}(1-F_\pi(p))\,dp",
            font_size=33,
            color=GREY_B,
        ).next_to(coi_eq, DOWN, aligned_edge=LEFT, buff=0.2)
        self.play(Write(survival))

        identity_1 = MathTex(
            r"\mathbb E[X]=\int_0^{\infty}\mathbb P(X>u)\,du\quad (X\ge 0)",
            font_size=31,
            color=GREY_B,
        ).next_to(survival, DOWN, aligned_edge=LEFT, buff=0.2)
        identity_2 = MathTex(
            r"X=P-\underline p,\;u=p-\underline p\Rightarrow\int_{\underline p}^{\bar p}(1-F_\pi(p))\,dp",
            font_size=31,
            color=GREY_B,
        ).next_to(identity_1, DOWN, aligned_edge=LEFT, buff=0.14)
        self.play(Write(identity_1))
        self.play(Write(identity_2))
        self.wait(1.0)


class COIOrderStatisticProofScene(Scene):
    def construct(self) -> None:
        title = scene_title("Why COI Erodes with Agent Saturation")
        self.play(Write(title))

        scope = MathTex(
            r"\text{Assumption: independent sessions; no shared quotes across agents}",
            font_size=22,
            color=GREY_B,
        ).next_to(title, DOWN, buff=0.16)
        self.play(FadeIn(scope, shift=DOWN * 0.06))

        key = MathTex(r"p_{(1)}=\min(p_1,\ldots,p_N)", font_size=42, color=HIGHLIGHT)
        key.next_to(scope, DOWN, buff=0.22)
        self.play(Write(key))
        self.play(Circumscribe(key, color=HIGHLIGHT, run_time=0.85))

        number_line = NumberLine(
            x_range=[P_MIN, P_MAX, 10],
            length=9.8,
            color=AXIS_INK,
            include_numbers=True,
            decimal_number_config={"num_decimal_places": 0, "color": INK},
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
            step_group = VGroup(dots, min_dot, min_tag)

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
        p2 = MathTex(r"\mathbb{P}(p_{(1)}>t)=[1-F(t)]^N", font_size=42, color=HIGHLIGHT)
        prob_group = VGroup(p1, p2).arrange(DOWN, aligned_edge=LEFT, buff=0.16)
        prob_group.to_edge(RIGHT).shift(UP * 0.75)

        self.play(Write(p1))
        self.play(Write(p2))

        cleanup_items: list = [key, number_line, floor_marker, floor_label]
        if current_group is not None:
            cleanup_items.append(current_group)
        if current_info is not None:
            cleanup_items.append(current_info)
        self.play(
            FadeOut(VGroup(*cleanup_items), shift=DOWN * 0.12),
            prob_group.animate.shift(UP * 0.26),
        )

        tail_axes = (
            Axes(
                x_range=[0, 1, 0.2],
                y_range=[0, 1, 0.2],
                x_length=4.1,
                y_length=2.45,
                tips=False,
                axis_config={"stroke_width": 2, "color": AXIS_INK},
            )
            .to_edge(RIGHT)
            .shift(DOWN * 1.0 + LEFT * 0.2)
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
        c_labels.next_to(tail_axes, UP, buff=0.08).align_to(tail_axes, RIGHT)
        tail_x = MathTex(r"F(t)", font_size=24).next_to(tail_axes, DOWN, buff=0.05)
        tail_y = MathTex(r"[1-F(t)]^N", font_size=24).next_to(
            tail_axes, LEFT, buff=0.05
        )

        self.play(FadeIn(tail_axes), Create(curve_1), Create(curve_4), Create(curve_16))
        self.play(FadeIn(c_labels), FadeIn(tail_x), FadeIn(tail_y))

        e1 = MathTex(
            r"\mathbb{E}[p_{(1)}]=\underline p+\int_{\underline p}^{\bar p}[1-F(t)]^N\,dt",
            font_size=32,
        )
        e2 = MathTex(
            r"X=p_{(1)}-\underline p\ge 0,\quad \mathbb E[X]=\int_0^{\infty}\mathbb P(X>u)\,du",
            font_size=27,
            color=GREY_B,
        )
        e3 = MathTex(
            r"\mathbb P(X>u)=\mathbb P\!\left(p_{(1)}>\underline p+u\right)=[1-F(\underline p+u)]^N",
            font_size=27,
            color=GREY_B,
        )
        e4 = MathTex(
            r"0\le[1-F(t)]^N\le1,\quad [1-F(t)]^N\to0\ \text{for } t>\underline p",
            font_size=27,
            color=GREY_B,
        )
        e5 = MathTex(
            r"\Rightarrow\ \lim_{N\to\infty}(\mathbb{E}[p_{(1)}]-\underline p)=0",
            font_size=38,
            color=HIGHLIGHT,
        )
        proof_block = VGroup(e1, e2, e3, e4, e5).arrange(
            DOWN, aligned_edge=LEFT, buff=0.12
        )
        proof_block.to_edge(LEFT).shift(UP * 0.45)
        self.play(Write(e1))
        self.play(Write(e2))
        self.play(Write(e3))
        self.play(Write(e4))
        self.play(Write(e5))

        conclusion = MathTex(
            r"N\to\infty\ \Rightarrow\ \mathbb{E}[p_{(1)}]-\underline p\to 0",
            font_size=28,
            color=GREY_B,
        )
        conclusion.to_edge(DOWN)
        self.play(FadeIn(conclusion, shift=UP * 0.1))
        self.play(Indicate(conclusion, color=HIGHLIGHT, run_time=0.9))
        self.wait(0.85)


class BehaviorKernelConstructionScene(Scene):
    def construct(self) -> None:
        title = scene_title("From Session Paths to Transition Kernels")
        self.play(Write(title))

        traj_h = MathTex(
            r"\tau_H:\ s_0\!\to s_1\!\to s_2\!\to s_3\!\to s_T",
            font_size=28,
            color=GREEN_C,
        )
        traj_a = MathTex(
            r"\tau_A:\ s_0\!\to s_1\!\to s_2\!\to s_1\!\to s_2\!\to\cdots",
            font_size=28,
            color=RED_C,
        )
        trajectories = VGroup(traj_h, traj_a).arrange(
            DOWN, aligned_edge=LEFT, buff=0.16
        )
        trajectories.next_to(title, DOWN, buff=0.45).align_to(title, LEFT)
        self.play(
            LaggedStart(
                *[FadeIn(t, shift=RIGHT * 0.2) for t in trajectories], lag_ratio=0.25
            )
        )

        mle = MathTex(
            r"\hat P(s'\mid s)=\frac{N(s,s')}{\sum_k N(s,k)}",
            font_size=40,
            color=HIGHLIGHT,
        )
        mle.next_to(trajectories, DOWN, aligned_edge=LEFT, buff=0.28)
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
            header_buff=0.4,
        )
        mats = (
            VGroup(counts, probs)
            .arrange(RIGHT, buff=0.95)
            .scale(0.92)
            .to_edge(DOWN)
            .shift(UP * 0.34)
        )
        arrow = Arrow(counts.get_right(), probs.get_left(), buff=0.18, stroke_width=4)
        arrow_tag = MathTex(
            r"\text{row-normalize}", font_size=20, color=GREY_B
        ).next_to(arrow, UP, buff=0.08)
        kernel_arrow = Arrow(
            mle.get_bottom(),
            mats.get_top() + UP * 0.05,
            buff=0.1,
            color=GREY_B,
            stroke_width=3.2,
        )
        self.play(
            FadeIn(mats, shift=UP * 0.12),
            FadeIn(arrow),
            FadeIn(arrow_tag),
            FadeIn(kernel_arrow, shift=DOWN * 0.06),
        )
        self.play(
            FadeOut(mle, shift=UP * 0.08),
            FadeOut(kernel_arrow, shift=DOWN * 0.08),
        )

        note = MathTex(
            r"\bar T_H,\bar T_A\ \text{feed KL and }\hat\alpha(\tau')",
            font_size=22,
            color=GREY_B,
        )
        note.next_to(mats, DOWN, buff=0.16)
        self.play(FadeIn(note, shift=UP * 0.1))
        self.wait(1.0)


class SeparabilitySignalScene(Scene):
    def construct(self) -> None:
        title = scene_title("Separability as Control Signal")
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

        self.play(
            kernels.animate.scale(0.6)
            .arrange(DOWN, aligned_edge=LEFT, buff=0.24)
            .to_edge(LEFT)
            .shift(UP * 0.18)
        )

        d_h = MathTex(r"\Delta_H=D_{KL}(\hat T'\parallel\bar T_H)", font_size=36)
        d_a = MathTex(r"\Delta_A=D_{KL}(\hat T'\parallel\bar T_A)", font_size=36)
        gap = MathTex(r"g=\Delta_H-\Delta_A", font_size=44, color=HIGHLIGHT)
        alpha = MathTex(r"\hat\alpha(\tau')=\sigma(\beta g)", font_size=40)
        eqs = VGroup(d_h, d_a, gap, alpha).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        eqs.to_edge(RIGHT).shift(UP * 0.38)
        self.play(LaggedStart(*[Write(eq) for eq in eqs], lag_ratio=0.18))
        self.play(Indicate(gap, color=HIGHLIGHT, run_time=0.85))

        self.play(
            eqs.animate.scale(0.66).next_to(kernels, DOWN, aligned_edge=LEFT, buff=0.16)
        )

        mu_h, sigma_h = -3.35, 2.67
        mu_a, sigma_a = 1.65, 2.83
        axis = (
            Axes(
                x_range=[-10, 10, 2],
                y_range=[0.0, 0.18, 0.03],
                x_length=6.8,
                y_length=3.7,
                tips=False,
                axis_config={"stroke_width": 2, "color": AXIS_INK},
            )
            .to_edge(RIGHT)
            .shift(DOWN * 0.75 + LEFT * 0.15)
        )
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
        h_label = Text("human", font_size=22, color=BLUE_D).move_to(
            axis.c2p(-6.4, 0.108)
        )
        a_label = Text("agent", font_size=22, color=RED_C).move_to(axis.c2p(5.8, 0.095))

        boundary = DashedLine(
            axis.c2p(0.0, 0.0), axis.c2p(0.0, 0.165), color=GREY_B, stroke_width=2
        )
        boundary_tag = Text("decision boundary", font_size=17, color=GREY_B).next_to(
            boundary, UP, buff=0.08
        )
        boundary_tag.shift(RIGHT * 0.8)

        g_obs = 1.6
        g_line = Line(
            axis.c2p(g_obs, 0.0),
            axis.c2p(g_obs, 0.145),
            color=HIGHLIGHT,
            stroke_width=4,
        )
        g_dot = Dot(axis.c2p(g_obs, 0.145), color=HIGHLIGHT, radius=0.06)
        g_tag = (
            MathTex(r"g_{obs}", color=HIGHLIGHT)
            .scale(0.72)
            .next_to(g_dot, UP, buff=0.04)
        )

        self.play(FadeIn(axis), FadeIn(x_tag))
        self.play(Create(human_curve), Create(agent_curve))
        self.play(
            FadeIn(h_label), FadeIn(a_label), FadeIn(boundary), FadeIn(boundary_tag)
        )
        self.play(FadeIn(g_line), FadeIn(g_dot), FadeIn(g_tag))

        hint = MathTex(
            r"g>0\ \Rightarrow\ \hat\alpha\ \text{upweights agent contamination}",
            font_size=22,
            color=GREY_B,
        )
        hint.next_to(x_tag, DOWN, buff=0.1)
        hint.match_x(axis)
        self.play(FadeIn(hint, shift=UP * 0.1))
        self.wait(1.0)


class ContaminationGeneratorScene(Scene):
    def construct(self) -> None:
        title = scene_title("Contamination Generator G(alpha)")
        self.play(Write(title))

        human_pool = card("labeled human sessions", color=BLUE_D, width=4.1)
        agent_pool = card("synthetic agent sessions", color=RED_C, width=4.1)
        mixed_pool = card("mixed batch for training", color=HIGHLIGHT, width=4.4)

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

        flow = VGroup(top, mixed_pool, a1, a2)
        self.play(flow.animate.scale(0.68).to_edge(LEFT).shift(UP * 0.58))

        alpha_tracker = ValueTracker(0.18)
        bar_outline = Rectangle(
            width=7.0, height=0.46, stroke_color=AXIS_INK, stroke_width=2
        ).move_to(RIGHT * 0.55 + DOWN * 0.12)
        base_h = Rectangle(
            width=7.0, height=0.4, stroke_width=0, fill_color=BLUE_D, fill_opacity=0.35
        ).move_to(bar_outline)

        def make_agent_fill() -> Rectangle:
            width = max(0.02, 7.0 * alpha_tracker.get_value())
            rect = Rectangle(
                width=width,
                height=0.4,
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
                color=HIGHLIGHT,
            ).next_to(alpha_label, RIGHT, buff=0.1)
        )
        left_tag = Text("human share (1-alpha)", font_size=18, color=BLUE_D).next_to(
            bar_outline, LEFT, buff=0.15
        )
        right_tag = Text("agent share (alpha)", font_size=18, color=RED_C).next_to(
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
            r"\hat Q(p\mid\tau')=(1-\alpha)\,\hat Q_H(p\mid\tau')+\alpha\,\hat Q_A(p\mid\tau')",
            font_size=31,
        ).next_to(bar_outline, DOWN, buff=0.45)
        interval = MathTex(
            r"\alpha\in[\alpha_0-\epsilon_\alpha,\,\alpha_0+\epsilon_\alpha]",
            font_size=31,
            color=GREY_B,
        )
        interval.next_to(mix_eq, DOWN, buff=0.2)
        self.play(Write(mix_eq), Write(interval))

        self.play(alpha_tracker.animate.set_value(0.32), run_time=1.2)
        self.play(alpha_tracker.animate.set_value(0.55), run_time=1.2)
        self.play(alpha_tracker.animate.set_value(0.24), run_time=1.1)
        self.wait(0.9)


class RobustControlScene(Scene):
    def construct(self) -> None:
        title = scene_title("Distributionally Robust Control Layer")
        self.play(Write(title))

        objective = MathTex(
            r"\pi^*=\arg\max_\pi\min_{Q\in\mathcal U_\epsilon}\mathbb E_{d\sim Q}[R(p,d)-\lambda\,\mathrm{COI}_{\mathrm{leak}}(p,\tau') ]",
            font_size=29,
        ).next_to(title, DOWN, buff=0.4)
        reward = MathTex(
            r"r_t=R(p_t,d_t)-\lambda f(\tau_t')c_{info},\quad d_t\sim Q(\cdot\mid p_t,\tau_t')",
            font_size=29,
            color=HIGHLIGHT,
        )
        reward.next_to(objective, DOWN, buff=0.22)
        demand_link = MathTex(
            r"\hat Q(p_t,\tau_t')=\mathbb E_Q[d_t\mid p_t,\tau_t']",
            font_size=27,
            color=GREY_B,
        ).next_to(reward, DOWN, buff=0.14)
        self.play(Write(objective), Write(reward), Write(demand_link))
        self.play(Circumscribe(objective, color=HIGHLIGHT, run_time=1.05))

        plane = (
            Axes(
                x_range=[-3, 3, 1],
                y_range=[-3, 3, 1],
                x_length=5.6,
                y_length=5.6,
                tips=False,
                axis_config={"stroke_width": 1.8, "color": AXIS_INK},
            )
            .to_edge(LEFT)
            .shift(DOWN * 0.55)
        )
        center = Dot(plane.c2p(0, 0), color=BLUE_D, radius=0.08)
        center_tag = (
            MathTex(r"\hat P_N", color=BLUE_D)
            .scale(0.75)
            .next_to(center, UP, buff=0.07)
        )
        ball = Circle(radius=1.75, color=HIGHLIGHT, stroke_width=3).move_to(center)
        ball_tag = (
            MathTex(r"\mathcal U_\epsilon", color=HIGHLIGHT)
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

        inner_step = card(
            "inner min picks Q*", color=RED_C, width=4.6, height=0.9, font_size=20
        )
        demand_step = card(
            "sample demand from Q*", color=ORANGE, width=4.6, height=0.9, font_size=20
        )
        update_step = card(
            "outer max updates policy",
            color=GREEN_C,
            width=4.6,
            height=0.9,
            font_size=20,
        )
        pipeline = (
            VGroup(inner_step, demand_step, update_step)
            .arrange(DOWN, buff=0.32)
            .to_edge(RIGHT)
            .shift(DOWN * 0.95)
        )
        chooser = Arrow(
            q2.get_right() + RIGHT * 0.15,
            inner_step.get_left(),
            buff=0.08,
            color=RED_C,
            stroke_width=4,
        )
        stage_arrow_1 = Arrow(
            inner_step.get_bottom(),
            demand_step.get_top(),
            buff=0.08,
            stroke_width=3.6,
        )
        stage_arrow_2 = Arrow(
            demand_step.get_bottom(),
            update_step.get_top(),
            buff=0.08,
            stroke_width=3.6,
        )
        feedback = CurvedArrow(
            update_step.get_left() + DOWN * 0.12,
            center.get_right() + UP * 0.15,
            angle=0.92,
            color=GREEN_C,
            stroke_width=3.6,
        )
        self.play(FadeIn(pipeline, shift=LEFT * 0.15))
        self.play(FadeIn(chooser))
        self.play(FadeIn(stage_arrow_1), FadeIn(stage_arrow_2))
        self.play(FadeIn(feedback))

        note = MathTex(
            r"r_t\ \text{uses }d_t\sim Q^\star\ \text{(inner min)}",
            font_size=24,
            color=GREY_B,
        )
        note.to_edge(DOWN)
        self.play(FadeIn(note, shift=UP * 0.1))
        self.wait(1.0)


class SystemLoopScene(Scene):
    def construct(self) -> None:
        title = scene_title("Online + Offline Defense Loop")
        self.play(Write(title))

        web = card("Web app", color=BLUE_D, width=2.9)
        provider = card("Pricing provider", color=BLUE_D, width=3.5)
        kafka = card("Kafka streams", color=HIGHLIGHT, width=3.1)
        kernels = card("Kernel + KL estimator", color=GREEN_C, width=3.9)
        generator = card("Generator G(alpha)", color=GREEN_C, width=3.5)
        policy = card("DR-RL trainer", color=ORANGE, width=3.0)

        web.move_to(LEFT * 4.6 + UP * 1.35)
        provider.move_to(RIGHT * 4.2 + UP * 1.35)
        kafka.move_to(LEFT * 4.6 + DOWN * 1.1)
        kernels.move_to(LEFT * 1.3 + DOWN * 1.1)
        generator.move_to(RIGHT * 2.0 + DOWN * 1.1)
        policy.move_to(RIGHT * 5.1 + DOWN * 1.1)

        online_tag = Text("online serving", font_size=22, weight="BOLD", color=GREY_B)
        online_tag.next_to(web, UP, buff=0.38).align_to(web, LEFT)
        offline_tag = Text(
            "offline defense training", font_size=22, weight="BOLD", color=GREY_B
        )
        offline_tag.next_to(kafka, UP, buff=0.38).align_to(kafka, LEFT)

        request_arrow = CurvedArrow(
            web.get_right() + UP * 0.2,
            provider.get_left() + UP * 0.2,
            angle=-0.24,
            stroke_width=4,
        )
        response_arrow = CurvedArrow(
            provider.get_left() + DOWN * 0.2,
            web.get_right() + DOWN * 0.2,
            angle=-0.24,
            stroke_width=4,
        )
        log_arrow = Arrow(web.get_bottom(), kafka.get_top(), buff=0.08, stroke_width=4)
        k_to_kl = Arrow(kafka.get_right(), kernels.get_left(), buff=0.1, stroke_width=4)
        kl_to_g = Arrow(
            kernels.get_right(), generator.get_left(), buff=0.1, stroke_width=4
        )
        g_to_pi = Arrow(
            generator.get_right(), policy.get_left(), buff=0.1, stroke_width=4
        )
        pi_to_provider = Arrow(
            policy.get_top(), provider.get_bottom(), buff=0.08, stroke_width=4
        )

        nodes = VGroup(web, provider, kafka, kernels, generator, policy)
        self.play(
            FadeIn(online_tag, shift=UP * 0.08), FadeIn(offline_tag, shift=UP * 0.08)
        )
        self.play(
            LaggedStart(
                *[FadeIn(node, shift=UP * 0.08) for node in nodes], lag_ratio=0.12
            )
        )
        self.play(
            LaggedStart(
                *[
                    FadeIn(a)
                    for a in [
                        request_arrow,
                        response_arrow,
                        log_arrow,
                        k_to_kl,
                        kl_to_g,
                        g_to_pi,
                        pi_to_provider,
                    ]
                ],
                lag_ratio=0.08,
            )
        )

        labels = VGroup(
            Text("request quote", font_size=17).next_to(request_arrow, UP, buff=0.06),
            Text("serve price", font_size=17).next_to(response_arrow, DOWN, buff=0.06),
            Text("events + quote logs", font_size=17).next_to(
                log_arrow, RIGHT, buff=0.08
            ),
            Text("fit kernels + alpha", font_size=17).next_to(kl_to_g, UP, buff=0.08),
            Text("robust policy train", font_size=17).next_to(g_to_pi, UP, buff=0.08),
            Text("publish model", font_size=17).next_to(
                pi_to_provider, RIGHT, buff=0.08
            ),
        )
        self.play(LaggedStart(*[FadeIn(l) for l in labels], lag_ratio=0.15))
        self.wait(1.0)


class ObjectiveAndResultsScene(Scene):
    def construct(self) -> None:
        title = scene_title("Experimental Signal (paired benchmarks)")
        self.play(Write(title))

        # Paired robust vs non-robust cohort; magnitudes align with sweep-scale logs.
        objective_chart = BarChart(
            values=[3.41, 3.91],
            bar_names=["robust", "non-robust"],
            y_range=[0, 5, 1],
            y_length=2.9,
            x_length=4.8,
            bar_colors=[GREEN_C, RED_C],
        )
        objective_label = MathTex(
            r"\text{objective}\ (\times 10^5)", font_size=22
        ).next_to(objective_chart, UP, buff=0.1)

        revenue_chart = BarChart(
            values=[3.80, 4.18],
            bar_names=["robust", "non-robust"],
            y_range=[0, 5, 1],
            y_length=2.9,
            x_length=4.8,
            bar_colors=[GREEN_C, RED_C],
        )
        revenue_label = MathTex(
            r"\text{revenue}\ (\times 10^5)", font_size=22
        ).next_to(revenue_chart, UP, buff=0.1)

        charts = VGroup(
            VGroup(objective_label, objective_chart),
            VGroup(revenue_label, revenue_chart),
        ).arrange(RIGHT, buff=0.85)
        charts.next_to(title, DOWN, buff=0.7)
        self.play(FadeIn(charts, shift=UP * 0.2))

        pairwise = MathTex(
            r"\textbf{wins:}\quad \tfrac{13}{40}\ (\text{obj}),\ \tfrac{16}{40}\ (\text{rev})",
            font_size=26,
        )
        pairwise.next_to(charts, DOWN, buff=0.32)
        self.play(FadeIn(pairwise, shift=RIGHT * 0.12))

        caution = MathTex(
            r"\text{regime-dependent};\ \text{read with COI + stability (Results)}",
            font_size=22,
            color=GREY_B,
        )
        caution.to_edge(DOWN)
        self.play(FadeIn(caution, shift=UP * 0.1))
        self.wait(1.1)


class ThesisBannerPosterScene(Scene):
    def construct(self) -> None:
        title = Text("PHANTOM", font_size=72, weight="BOLD", color=INK).to_edge(UP)
        subtitle = Text(
            "Pricing Heuristics Against Non-human Transaction Orchestration",
            font_size=24,
            color=GREY_B,
        ).next_to(title, DOWN, buff=0.05)

        coi_axes = Axes(
            x_range=[0, 1, 0.2],
            y_range=[0, 1, 0.2],
            x_length=3.15,
            y_length=1.75,
            tips=False,
            axis_config={"stroke_width": 1.8, "color": AXIS_INK},
        )
        coi_n1 = coi_axes.plot(
            lambda x: (1 - x) ** 1,
            x_range=[0, 1],
            color=BLUE_D,
            stroke_width=4,
        )
        coi_n8 = coi_axes.plot(
            lambda x: (1 - x) ** 8,
            x_range=[0, 1],
            color=ORANGE,
            stroke_width=4,
        )
        coi_hint = Text(
            "Order-statistic tail compresses as query count grows", font_size=15
        )
        coi_hint.set_color(GREY_B).next_to(coi_axes, DOWN, buff=0.06)
        coi_title = Text("1) COI erosion", font_size=23, weight="BOLD", color=ORANGE)
        coi_body = VGroup(coi_axes, coi_n1, coi_n8, coi_hint)
        coi_group = VGroup(coi_title, coi_body).arrange(DOWN, buff=0.08)
        coi_frame = SurroundingRectangle(coi_group, color=ORANGE, buff=0.14)
        coi_frame.set_fill(color=ORANGE, opacity=0.05)
        coi_panel = VGroup(coi_frame, coi_group)

        gap_axes = Axes(
            x_range=[-8, 8, 2],
            y_range=[0.0, 0.2, 0.05],
            x_length=3.15,
            y_length=1.75,
            tips=False,
            axis_config={"stroke_width": 1.8, "color": AXIS_INK},
        )
        gap_h = gap_axes.plot(
            lambda x: normal_pdf(x, -3.35, 2.67),
            x_range=[-8, 8],
            color=BLUE_D,
            stroke_width=4,
        )
        gap_a = gap_axes.plot(
            lambda x: normal_pdf(x, 1.65, 2.83),
            x_range=[-8, 8],
            color=RED_C,
            stroke_width=4,
        )
        gap_boundary = DashedLine(
            gap_axes.c2p(0, 0),
            gap_axes.c2p(0, 0.17),
            color=GREY_B,
            stroke_width=2,
        )
        gap_hint = Text(
            "Gap score g = Delta_H - Delta_A drives alpha-hat", font_size=15
        )
        gap_hint.set_color(GREY_B).next_to(gap_axes, DOWN, buff=0.06)
        gap_title = Text(
            "2) Behavioral separability", font_size=23, weight="BOLD", color=GREEN_C
        )
        gap_body = VGroup(gap_axes, gap_h, gap_a, gap_boundary, gap_hint)
        gap_group = VGroup(gap_title, gap_body).arrange(DOWN, buff=0.08)
        gap_frame = SurroundingRectangle(gap_group, color=GREEN_C, buff=0.14)
        gap_frame.set_fill(color=GREEN_C, opacity=0.05)
        gap_panel = VGroup(gap_frame, gap_group)

        ctrl_title = Text(
            "3) Robust pricing control", font_size=23, weight="BOLD", color=HIGHLIGHT
        )
        ctrl_signal = MathTex(r"\hat\alpha(\tau')=\sigma(\beta g)", font_size=31)
        ctrl_policy = MathTex(
            r"\pi^*=\arg\max_\pi\min_{Q\in\mathcal U_\epsilon}\mathbb E[r]",
            font_size=29,
            color=HIGHLIGHT,
        )
        ctrl_steps = VGroup(
            card(
                "estimate contamination from behavior",
                color=GREEN_C,
                width=4.0,
                height=0.72,
                font_size=16,
            ),
            card(
                "optimize price policy under uncertainty",
                color=ORANGE,
                width=4.0,
                height=0.72,
                font_size=16,
            ),
        ).arrange(DOWN, buff=0.18)
        ctrl_arrow = Arrow(
            ctrl_steps[0].get_bottom(),
            ctrl_steps[1].get_top(),
            buff=0.06,
            color=AXIS_INK,
            stroke_width=3,
        )
        ctrl_body = VGroup(ctrl_signal, ctrl_policy, ctrl_steps, ctrl_arrow).arrange(
            DOWN, buff=0.14
        )
        ctrl_group = VGroup(ctrl_title, ctrl_body).arrange(DOWN, buff=0.08)
        ctrl_frame = SurroundingRectangle(ctrl_group, color=HIGHLIGHT, buff=0.14)
        ctrl_frame.set_fill(color=HIGHLIGHT, opacity=0.05)
        ctrl_panel = VGroup(ctrl_frame, ctrl_group)

        panels = VGroup(coi_panel, gap_panel, ctrl_panel).arrange(RIGHT, buff=0.3)
        panels.scale(0.92).next_to(subtitle, DOWN, buff=0.28)

        web = card("web sessions", color=BLUE_D, width=2.2, height=0.7, font_size=17)
        kafka = card(
            "quote + event logs", color=HIGHLIGHT, width=2.6, height=0.7, font_size=17
        )
        kernel = card(
            "transition kernels", color=GREEN_C, width=2.5, height=0.7, font_size=17
        )
        policy = card(
            "robust policy", color=ORANGE, width=2.2, height=0.7, font_size=17
        )
        flow_nodes = VGroup(web, kafka, kernel, policy).arrange(RIGHT, buff=0.22)
        flow_nodes.to_edge(DOWN, buff=0.52)
        flow_arrows = VGroup(
            Arrow(web.get_right(), kafka.get_left(), buff=0.05, stroke_width=2.8),
            Arrow(kafka.get_right(), kernel.get_left(), buff=0.05, stroke_width=2.8),
            Arrow(kernel.get_right(), policy.get_left(), buff=0.05, stroke_width=2.8),
        )

        status = VGroup(
            Text("Mann-Whitney p = 0.0006", font_size=19, color=GREEN_C),
            Text("Pairwise robust wins: 13/40 objective, 16/40 revenue", font_size=19),
        ).arrange(DOWN, buff=0.06)
        status[1].set_color(GREY_B)
        status.next_to(flow_nodes, UP, buff=0.15)

        footer = Text(
            "From mechanism failure to an implementable defense loop",
            font_size=25,
            color=HIGHLIGHT,
        ).next_to(flow_nodes, DOWN, buff=0.13)

        self.add(
            title,
            subtitle,
            panels,
            flow_nodes,
            flow_arrows,
            status,
            footer,
        )
        self.wait(0.1)


class RewardAndLeakageScene(Scene):
    def construct(self) -> None:
        title = scene_title("Reward: Revenue vs COI Leakage")
        self.play(Write(title))

        leak = MathTex(
            r"\mathrm{COI}_{\mathrm{leak}}(p,\tau')=f(\tau')\cdot\mathrm{InfoValue}(p,\tau')}",
            font_size=34,
            color=HIGHLIGHT,
        ).next_to(title, DOWN, buff=0.42)
        self.play(Write(leak))
        self.play(Circumscribe(leak, color=HIGHLIGHT, run_time=1.1))

        info = MathTex(
            r"\mathrm{InfoValue}\in\{1,\,-\log\pi(p\mid\tau')\}",
            font_size=30,
            color=GREY_B,
        ).next_to(leak, DOWN, buff=0.22)
        self.play(Write(info))

        f_track = ValueTracker(0.0)
        bar_w, bar_h = 4.2, 0.38
        bar_bg = Rectangle(
            width=bar_w,
            height=bar_h,
            stroke_color=AXIS_INK,
            stroke_width=2,
            fill_opacity=0.06,
        ).shift(DOWN * 0.75)

        def leak_bar() -> Rectangle:
            w = max(0.04, bar_w * float(f_track.get_value()))
            r = Rectangle(
                width=w,
                height=bar_h - 0.06,
                stroke_width=0,
                fill_color=RED_C,
                fill_opacity=0.55,
            )
            r.move_to(bar_bg.get_left() + RIGHT * (w / 2.0) + RIGHT * 0.02)
            return r

        leak_fill = always_redraw(leak_bar)
        scale_lbl = MathTex(
            r"\lambda\cdot\mathrm{COI}_{\mathrm{leak}}",
            font_size=28,
        ).next_to(bar_bg, UP, buff=0.18)
        f_readout = always_redraw(
            lambda: MathTex(
                rf"f(\tau')={f_track.get_value():.2f}",
                font_size=32,
                color=HIGHLIGHT,
            ).next_to(bar_bg, DOWN, buff=0.2)
        )

        self.play(FadeIn(bar_bg), Write(scale_lbl))
        self.add(leak_fill, f_readout)
        self.play(f_track.animate.set_value(0.88), run_time=2.4, rate_func=smooth)
        self.wait(0.25)

        objective = MathTex(
            r"\pi^*=\arg\max_\pi\min_{Q\in\mathcal{U}_\epsilon}\mathbb{E}_{d\sim Q}\Big["
            r"R(p,d)-\lambda\,\mathrm{COI}_{\mathrm{leak}}(p,\tau')\Big]",
            font_size=28,
        ).to_edge(DOWN, buff=0.32)
        self.play(Write(objective))
        self.play(Indicate(objective, color=HIGHLIGHT, run_time=1.0))
        self.wait(0.7)


class StackelbergAmbiguityScene(Scene):
    def construct(self) -> None:
        title = scene_title("Stackelberg Step + Contamination Ambiguity")
        self.play(Write(title))

        stack = VGroup(
            MathTex(r"\text{leader: platform chooses }p_t", font_size=32),
            MathTex(
                r"\text{follower: sessions }(d_t,\tau_t')\text{ under }Q(\cdot\mid p_t,\tau_t')",
                font_size=28,
                color=GREY_B,
            ),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        stack.next_to(title, DOWN, buff=0.38).align_to(title, LEFT)
        self.play(LaggedStart(*[Write(line) for line in stack], lag_ratio=0.2))

        nl = NumberLine(
            x_range=[0, 1, 0.2],
            length=7.2,
            color=AXIS_INK,
            include_numbers=True,
            decimal_number_config={"num_decimal_places": 1, "color": GREY_B},
        ).shift(DOWN * 0.55)

        alpha0, eps_a = 0.35, 0.12
        lo, hi = max(0.0, alpha0 - eps_a), min(1.0, alpha0 + eps_a)

        self.play(Create(nl))
        tick0 = Line(
            nl.n2p(alpha0) + UP * 0.12,
            nl.n2p(alpha0) + DOWN * 0.12,
            color=HIGHLIGHT,
            stroke_width=4,
        )
        interval = Line(
            nl.n2p(lo),
            nl.n2p(hi),
            color=HIGHLIGHT,
            stroke_width=10,
        )
        self.play(Create(tick0), Create(interval))

        amb = MathTex(
            r"\mathcal{A}_{\epsilon_\alpha}(\alpha_0)=\{\alpha:\lvert\alpha-\alpha_0\rvert\le\epsilon_\alpha\}",
            font_size=30,
        ).next_to(nl, DOWN, buff=0.42)
        self.play(Write(amb))
        self.play(Circumscribe(interval, color=HIGHLIGHT, run_time=0.9))
        self.wait(0.6)


SCENE_ORDER = [
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

POSTER_SCENES = ["ThesisBannerPosterScene"]
