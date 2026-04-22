from __future__ import annotations

import numpy as np
from manim import *
from common import AXIS_INK, HIGHLIGHT, INK, P_MAX, P_MIN, card, normal_pdf, scene_title, to_matrix


class LogsToKernelsScene(Scene):
    def construct(self):
        title = scene_title("From Event Logs to Transition Kernels")
        self.play(Write(title))

        # 1. Logs
        log_lines = VGroup(
            Text('{"session": "H1", "event": "start"}', font="monospace", font_size=16),
            Text('{"session": "A1", "event": "start"}', font="monospace", font_size=16),
            Text('{"session": "H1", "event": "view"}', font="monospace", font_size=16),
            Text('{"session": "A1", "event": "view"}', font="monospace", font_size=16),
            Text(
                '{"session": "H1", "event": "detail"}', font="monospace", font_size=16
            ),
            Text(
                '{"session": "A1", "event": "detail"}', font="monospace", font_size=16
            ),
            Text('{"session": "H1", "event": "cart"}', font="monospace", font_size=16),
            Text('{"session": "A1", "event": "view"}', font="monospace", font_size=16),
            Text('{"session": "H1", "event": "buy"}', font="monospace", font_size=16),
            Text(
                '{"session": "A1", "event": "detail"}', font="monospace", font_size=16
            ),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        log_lines.to_edge(LEFT, buff=1.0).shift(UP * 0.5)

        self.play(
            LaggedStart(
                *[FadeIn(line, shift=UP * 0.1) for line in log_lines], lag_ratio=0.1
            )
        )
        self.wait(0.5)

        # 2. Nodes in a grid
        def create_node(text, color):
            circ = Circle(radius=0.4, color=color, fill_opacity=0.2)
            lbl = Text(text, font_size=14).move_to(circ)
            return VGroup(circ, lbl)

        h_states = ["start", "view", "detail", "cart", "buy"]
        a_states = ["start", "view", "detail", "view", "detail"]

        h_nodes = VGroup(*[create_node(s, BLUE_D) for s in h_states]).arrange(
            RIGHT, buff=0.5
        )
        a_nodes = VGroup(*[create_node(s, RED_C) for s in a_states]).arrange(
            RIGHT, buff=0.5
        )

        trajectories = VGroup(h_nodes, a_nodes).arrange(DOWN, buff=1.0)
        trajectories.to_edge(RIGHT, buff=1.0).shift(UP * 0.5)

        h_label = Text("Human Trajectory", font_size=18, color=BLUE_D).next_to(
            h_nodes, UP
        )
        a_label = Text("Agent Trajectory", font_size=18, color=RED_C).next_to(
            a_nodes, UP
        )

        self.play(
            ReplacementTransform(log_lines[0::2], h_nodes),
            ReplacementTransform(log_lines[1::2], a_nodes),
            FadeIn(h_label),
            FadeIn(a_label),
        )

        # Add connecting lines
        h_lines = VGroup(
            *[
                Line(h_nodes[i].get_right(), h_nodes[i + 1].get_left(), color=BLUE_D)
                for i in range(len(h_nodes) - 1)
            ]
        )
        a_lines = VGroup(
            *[
                Line(a_nodes[i].get_right(), a_nodes[i + 1].get_left(), color=RED_C)
                for i in range(len(a_nodes) - 1)
            ]
        )

        self.play(Create(h_lines), Create(a_lines))
        self.wait(1)

        # 3. Counts to Kernel
        mle_text = MathTex(
            r"\hat P(s'\mid s) = \frac{N(s,s')}{\sum_k N(s,k)}",
            font_size=36,
            color=HIGHLIGHT,
        )
        mle_text.next_to(trajectories, DOWN, buff=0.8)
        self.play(Write(mle_text))

        counts = to_matrix(
            [
                [0, 8, 0, 0],
                [0, 2, 5, 1],
                [0, 3, 2, 4],
                [0, 1, 0, 6],
            ],
            "Count Matrix N",
            color=BLUE_D,
            fmt=".0f",
        )

        probs = to_matrix(
            [
                [0.00, 1.00, 0.00, 0.00],
                [0.00, 0.25, 0.62, 0.13],
                [0.00, 0.33, 0.22, 0.45],
                [0.00, 0.14, 0.00, 0.86],
            ],
            "Kernel T",
            color=GREEN_C,
        )

        mats = VGroup(counts, probs).arrange(RIGHT, buff=1.5).scale(0.65)

        arrow = Arrow(counts.get_right(), probs.get_left(), buff=0.2)
        arrow_lbl = MathTex(
            r"\text{normalize}", font_size=18, color=GREY_B
        ).next_to(arrow, UP)

        # clear top half to make space if needed
        self.play(
            FadeOut(h_nodes),
            FadeOut(a_nodes),
            FadeOut(h_lines),
            FadeOut(a_lines),
            FadeOut(h_label),
            FadeOut(a_label),
            mle_text.animate.to_edge(UP, buff=1.5).set_x(0),
        )
        mats.next_to(mle_text, DOWN, buff=0.5)
        arrow.move_to((counts.get_right() + probs.get_left()) / 2)
        arrow_lbl.next_to(arrow, UP)

        self.play(FadeIn(counts, shift=UP * 0.2))
        self.play(GrowArrow(arrow), FadeIn(arrow_lbl))
        self.play(FadeIn(probs, shift=UP * 0.2))
        self.wait(1)


class KLSeparabilityAndSignificanceScene(Scene):
    def construct(self):
        title = scene_title("Behavioral Separability & Significance")
        self.play(Write(title))

        human_mat = to_matrix(
            [
                [0.05, 0.70, 0.20, 0.05],
                [0.05, 0.20, 0.60, 0.15],
                [0.10, 0.25, 0.30, 0.35],
                [0.00, 0.00, 0.00, 1.00],
            ],
            "Human Centroid T_H",
            BLUE_D,
        ).scale(0.7)

        agent_mat = to_matrix(
            [
                [0.03, 0.82, 0.12, 0.03],
                [0.06, 0.55, 0.21, 0.18],
                [0.08, 0.48, 0.14, 0.30],
                [0.00, 0.00, 0.00, 1.00],
            ],
            "Agent Centroid T_A",
            RED_C,
        ).scale(0.7)

        centroids = VGroup(human_mat, agent_mat).arrange(RIGHT, buff=1.0)
        centroids.next_to(title, DOWN, buff=0.5)
        self.play(FadeIn(centroids, shift=DOWN * 0.2))

        # Trajectory
        t_prime = MathTex(r"\hat T'", font_size=36, color=HIGHLIGHT)
        d_h = MathTex(r"\Delta_H = D_{KL}(\hat T' \parallel \bar T_H)", font_size=32)
        d_a = MathTex(r"\Delta_A = D_{KL}(\hat T' \parallel \bar T_A)", font_size=32)
        gap = MathTex(r"g = \Delta_H - \Delta_A", font_size=36, color=HIGHLIGHT)

        eqs = VGroup(t_prime, d_h, d_a, gap).arrange(DOWN, buff=0.2)
        eqs.to_edge(LEFT, buff=1.0).shift(DOWN * 1.0)

        self.play(Write(eqs))

        # Distributions
        axis = (
            Axes(
                x_range=[-8, 8, 2],
                y_range=[0, 0.2, 0.05],
                x_length=6,
                y_length=3,
                tips=False,
                axis_config={"color": AXIS_INK, "stroke_width": 2},
            )
            .to_edge(RIGHT, buff=1.0)
            .shift(DOWN * 1.0)
        )

        mu_h, sig_h = -3.5, 2.0
        mu_a, sig_a = 3.5, 2.0

        h_curve = axis.plot(
            lambda x: normal_pdf(x, mu_h, sig_h), color=BLUE_D, stroke_width=4
        )
        a_curve = axis.plot(
            lambda x: normal_pdf(x, mu_a, sig_a), color=RED_C, stroke_width=4
        )

        h_lbl = (
            Text("Human", color=BLUE_D, font_size=20)
            .next_to(h_curve, UP, buff=-0.5)
            .shift(LEFT * 1)
        )
        a_lbl = (
            Text("Agent", color=RED_C, font_size=20)
            .next_to(a_curve, UP, buff=-0.5)
            .shift(RIGHT * 1)
        )

        boundary = DashedLine(axis.c2p(0, 0), axis.c2p(0, 0.18), color=GREY_B)

        self.play(FadeIn(axis))
        self.play(Create(h_curve), Create(a_curve))
        self.play(FadeIn(h_lbl), FadeIn(a_lbl), FadeIn(boundary))

        sig_text = MathTex(
            r"p<10^{-3}\ \text{(Mann--Whitney)}", font_size=24, color=GREEN_C
        )
        sig_text.next_to(axis, DOWN, buff=0.3)
        self.play(Write(sig_text))
        self.wait(1)


class TrajectorySamplingScene(Scene):
    def construct(self):
        title = scene_title("Generative Trajectory Sampling")
        self.play(Write(title))

        agent_mat = to_matrix(
            [
                [0.00, 0.80, 0.20, 0.00, 0.00],
                [0.00, 0.30, 0.50, 0.20, 0.00],
                [0.00, 0.40, 0.30, 0.30, 0.00],
                [0.00, 0.10, 0.10, 0.10, 0.70],
                [0.00, 0.00, 0.00, 0.00, 1.00],
            ],
            "Agent Kernel T_A",
            RED_C,
        ).scale(0.6)
        agent_mat.to_edge(LEFT, buff=1.0)

        self.play(FadeIn(agent_mat))

        states = ["Start", "View", "Detail", "Cart", "Buy"]

        def create_node(text):
            circ = Circle(radius=0.4, color=AXIS_INK, fill_opacity=0.1)
            lbl = Text(text, font_size=16).move_to(circ)
            return VGroup(circ, lbl)

        nodes = VGroup(*[create_node(s) for s in states]).arrange(RIGHT, buff=0.6)
        nodes.to_edge(RIGHT, buff=0.5).shift(UP * 1.0)

        self.play(FadeIn(nodes))

        # Output trajectory string
        traj_label = (
            Text("Sampled Trajectory:", font_size=24, color=HIGHLIGHT)
            .to_edge(DOWN)
            .shift(UP * 1.5 + LEFT * 1)
        )
        self.play(FadeIn(traj_label))

        walker = Dot(color=HIGHLIGHT, radius=0.15)
        walker.move_to(nodes[0].get_top() + UP * 0.2)

        self.play(FadeIn(walker))

        # Simulation
        path = [0, 1, 2, 1, 2]  # Start -> View -> Detail -> View -> Detail

        # We will build the string
        current_traj = VGroup(Text("Start", font_size=24, color=RED_C)).next_to(
            traj_label, RIGHT
        )
        self.play(FadeIn(current_traj))

        for i in range(len(path) - 1):
            curr_state = path[i]
            next_state = path[i + 1]

            # highlight row
            mat_core = agent_mat[2]  # the matrix itself

            # Using get_rows() which is standard in Mobject Matrix
            row_entries = mat_core.get_rows()[curr_state]
            row_rect = SurroundingRectangle(row_entries, color=HIGHLIGHT, buff=0.1)
            self.play(Create(row_rect), run_time=0.5)

            # move walker
            arc = CurvedArrow(
                walker.get_center(),
                nodes[next_state].get_top() + UP * 0.2,
                angle=-TAU / 4,
            )
            self.play(MoveAlongPath(walker, arc), run_time=1.0)

            # Update string
            arrow_str = MathTex(r"\rightarrow", font_size=24).next_to(
                current_traj, RIGHT
            )
            next_str = Text(states[next_state], font_size=24, color=RED_C).next_to(
                arrow_str, RIGHT
            )

            self.play(
                FadeIn(arrow_str), FadeIn(next_str), FadeOut(row_rect), run_time=0.5
            )
            current_traj.add(arrow_str, next_str)

        self.wait(1)


class KroneckerExpansionScene(Scene):
    def construct(self):
        title = scene_title("State-Space Expansion")
        self.play(Write(title))

        t_mat = to_matrix([[0.2, 0.8], [0.4, 0.6]], "Behavior T", BLUE_D)

        d_mat = to_matrix([[0.9, 0.1], [0.5, 0.5]], "Demand D", RED_C)

        kron_sym = MathTex(r"\otimes", font_size=60)
        eq_sym = MathTex(r"=", font_size=60)

        lhs = VGroup(t_mat, kron_sym, d_mat).arrange(RIGHT, buff=0.5)
        lhs.next_to(title, DOWN, buff=1.0)

        self.play(FadeIn(t_mat), FadeIn(d_mat), Write(kron_sym))
        self.wait(1)

        self.play(lhs.animate.scale(0.6).to_edge(LEFT, buff=0.5))

        # Show expanded
        # T tensor D
        expanded = to_matrix(
            [
                [0.18, 0.02, 0.72, 0.08],
                [0.10, 0.10, 0.40, 0.40],
                [0.36, 0.04, 0.54, 0.06],
                [0.20, 0.20, 0.30, 0.30],
            ],
            r"Expanded P = T \otimes D",
            HIGHLIGHT,
        ).scale(0.6)

        eq_sym.next_to(lhs, RIGHT, buff=0.5)
        expanded.next_to(eq_sym, RIGHT, buff=0.5)

        self.play(Write(eq_sym), FadeIn(expanded, shift=LEFT * 0.5))

        # Highlight a block
        # the top right block (0.8 * D)
        # rows 0,1 cols 2,3
        # In expanded:
        # row 0: 0, 1, 2, 3
        # row 1: 4, 5, 6, 7
        t_entries = t_mat[2].get_entries()
        if len(t_entries) >= 2:
            rect_T = SurroundingRectangle(
                t_entries[1], color=HIGHLIGHT
            )  # T[0,1] is 0.8
        else:
            rect_T = VGroup()

        exp_entries = expanded[2].get_entries()
        if len(exp_entries) >= 8:
            block_entries = VGroup(
                exp_entries[2], exp_entries[3], exp_entries[6], exp_entries[7]
            )
            rect_block = SurroundingRectangle(block_entries, color=HIGHLIGHT)
        else:
            rect_block = VGroup()

        desc = MathTex(
            r"P(s', d' \mid s, d)=T(s'\mid s)\,D(d'\mid d, s')",
            font_size=26,
            color=HIGHLIGHT,
        )
        desc.next_to(expanded, DOWN, buff=0.5)

        if len(t_entries) >= 2 and len(exp_entries) >= 8:
            self.play(Create(rect_T), Create(rect_block))
        self.play(Write(desc))
        self.wait(1)


class SamplingAndReservationScene(Scene):
    def construct(self):
        title = scene_title("Pricing Policy & Reservation Price")
        self.play(Write(title))

        # 1. The setup
        setup = VGroup(
            MathTex(r"p_i \sim \pi(p \mid \tau)", font_size=44),
            MathTex(
                r"\underline p = \text{reservation price}", font_size=38, color=ORANGE
            ),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        setup.to_edge(LEFT, buff=1.0).shift(UP * 1.0)

        self.play(Write(setup[0]))
        self.play(Write(setup[1]))

        # 2. Number line sampling
        number_line = NumberLine(
            x_range=[P_MIN, P_MAX, 10],
            length=9.8,
            color=AXIS_INK,
            include_numbers=True,
            decimal_number_config={"num_decimal_places": 0, "color": INK},
        ).shift(DOWN * 1.0)

        self.play(FadeIn(number_line))

        # Floor marker
        floor_marker = Line(
            number_line.n2p(P_MIN),
            number_line.n2p(P_MIN) + UP * 0.85,
            color=ORANGE,
            stroke_width=5,
        )
        floor_label = MathTex(r"\underline p", color=ORANGE).next_to(
            floor_marker, UP, buff=0.05
        )
        self.play(Create(floor_marker), FadeIn(floor_label))

        # Animate sampling
        rng = np.random.default_rng(42)
        n_samples = 5
        draws = np.sort(rng.beta(2.5, 2.0, size=n_samples) * (P_MAX - P_MIN) + P_MIN)

        dots = VGroup()
        for i, val in enumerate(draws):
            # Show drawing process
            temp_dot = Dot(number_line.n2p(120), radius=0.08, color=BLUE_D).shift(
                UP * 1.5
            )
            self.play(FadeIn(temp_dot), run_time=0.2)

            final_pos = number_line.n2p(float(val))
            self.play(temp_dot.animate.move_to(final_pos), run_time=0.3)
            dots.add(temp_dot)

        self.wait(0.5)

        # Highlight minimum
        min_dot = dots[0]
        min_highlight = Circle(radius=0.15, color=RED_C).move_to(min_dot)
        min_tag = MathTex(r"p_{(1)}", color=RED_C).next_to(min_highlight, UP, buff=0.1)

        self.play(Create(min_highlight), Write(min_tag))

        desc = MathTex(
            r"\text{realized price }p_{(1)}=\min\{p_1,\ldots,p_N\}",
            font_size=26,
            color=GREY_B,
        ).to_edge(DOWN)

        self.play(FadeIn(desc, shift=UP * 0.2))
        self.wait(1.5)


class COIDistributionScene(Scene):
    def construct(self):
        title = scene_title("Cost of Information (COI)")
        self.play(Write(title))

        # COI definition
        coi_def = MathTex(
            r"\mathrm{COI} = \mathbb{E}[P] - \underline p",
            font_size=46,
            color=HIGHLIGHT,
        ).next_to(title, DOWN, buff=0.5)

        self.play(Write(coi_def))

        # Distribution plot
        floor_x = 86.0
        mean_x = 116.0
        axes = Axes(
            x_range=[80, 160, 10],
            y_range=[0.0, 0.04, 0.01],
            x_length=8.0,
            y_length=4.0,
            tips=False,
            axis_config={"stroke_width": 2, "color": AXIS_INK},
        ).shift(DOWN * 0.5)

        density = axes.plot(
            lambda x: normal_pdf(x, mean_x, 12.0),
            x_range=[80, 160],
            color=BLUE_D,
            stroke_width=6,
        )

        area = axes.get_area(density, x_range=[80, 160], color=BLUE_D, opacity=0.2)

        self.play(FadeIn(axes))
        self.play(Create(density), FadeIn(area))

        # Markers
        floor_line = DashedLine(
            axes.c2p(floor_x, 0.0),
            axes.c2p(floor_x, 0.038),
            color=ORANGE,
            stroke_width=4,
        )
        mean_line = DashedLine(
            axes.c2p(mean_x, 0.0),
            axes.c2p(mean_x, 0.038),
            color=GREEN_C,
            stroke_width=4,
        )

        floor_tag = MathTex(r"\underline p", color=ORANGE).next_to(
            floor_line, UP, buff=0.1
        )
        mean_tag = MathTex(r"\mathbb{E}[P]", color=GREEN_C).next_to(
            mean_line, UP, buff=0.1
        )

        self.play(Create(floor_line), Write(floor_tag))
        self.play(Create(mean_line), Write(mean_tag))

        # COI span
        coi_arrow = DoubleArrow(
            axes.c2p(floor_x, 0.02), axes.c2p(mean_x, 0.02), color=HIGHLIGHT, buff=0
        )
        coi_label = Text("COI", font_size=24, color=HIGHLIGHT).next_to(
            coi_arrow, UP, buff=0.1
        )

        self.play(GrowFromCenter(coi_arrow), Write(coi_label))

        desc = MathTex(
            r"\mathrm{COI}=\mathbb{E}[P]-\underline p",
            font_size=28,
            color=GREY_B,
        ).to_edge(DOWN)

        self.play(FadeIn(desc, shift=UP * 0.2))
        self.wait(1.5)


class COIErosionMathScene(Scene):
    def construct(self):
        title = scene_title("Mathematical Proof of COI Erosion")
        self.play(Write(title))

        # Step 1: Expected value of minimum
        eq1 = MathTex(
            r"\mathbb{E}[p_{(1)}] = \underline p + \int_{\underline p}^{\bar p} \mathbb{P}(p_{(1)} > t) dt",
            font_size=36,
        )

        # Step 2: Probability of minimum > t
        eq2 = MathTex(
            r"\mathbb{P}(p_{(1)} > t) = \mathbb{P}(p_1 > t) \times \dots \times \mathbb{P}(p_N > t)",
            font_size=36,
        )

        # Step 3: Assuming i.i.d
        eq3 = MathTex(r"= [1 - F_\pi(t)]^N", font_size=36, color=HIGHLIGHT)

        # Step 4: Substitute back
        eq4 = MathTex(
            r"\mathbb{E}[p_{(1)}] = \underline p + \int_{\underline p}^{\bar p} [1 - F_\pi(t)]^N dt",
            font_size=36,
        )

        # Step 5: Limit as N -> inf
        eq5_pt1 = MathTex(
            r"\text{Since } [1 - F_\pi(t)] < 1 \text{ for } t > \underline p:",
            font_size=32,
            color=GREY_B,
        )

        eq5_pt2 = MathTex(
            r"\lim_{N \to \infty} \mathbb{E}[p_{(1)}] = \underline p",
            font_size=42,
            color=RED_C,
        )

        eq6 = MathTex(
            r"\lim_{N \to \infty} \mathrm{COI} = 0", font_size=46, color=HIGHLIGHT
        )

        group = VGroup(eq1, eq2, eq3, eq4, eq5_pt1, eq5_pt2, eq6).arrange(
            DOWN, aligned_edge=LEFT, buff=0.4
        )
        group.next_to(title, DOWN, buff=0.5).shift(RIGHT * 1.5)

        # We want eq3 to be right after eq2
        eq3.next_to(eq2, RIGHT, buff=0.2)

        # Re-arrange carefully
        step1 = eq1.copy().to_edge(LEFT, buff=1.0).shift(UP * 1.5)
        step2 = (
            VGroup(eq2.copy(), eq3.copy())
            .arrange(RIGHT, buff=0.2)
            .next_to(step1, DOWN, aligned_edge=LEFT, buff=0.5)
        )
        step3 = eq4.copy().next_to(step2, DOWN, aligned_edge=LEFT, buff=0.5)

        step4_group = (
            VGroup(eq5_pt1.copy(), eq5_pt2.copy())
            .arrange(DOWN, aligned_edge=LEFT, buff=0.2)
            .next_to(step3, DOWN, aligned_edge=LEFT, buff=0.5)
        )

        step5 = eq6.copy().next_to(step4_group, DOWN, buff=0.6).match_x(title)

        # Animate
        self.play(Write(step1))
        self.wait(0.5)

        self.play(Write(step2[0]))
        self.play(Write(step2[1]))
        self.wait(0.5)

        self.play(Write(step3))
        self.wait(0.5)

        self.play(Write(step4_group[0]))
        self.play(Write(step4_group[1]))
        self.wait(0.5)

        # Put a box around the final conclusion
        box = SurroundingRectangle(step5, color=HIGHLIGHT, buff=0.2)
        self.play(Write(step5), Create(box))

        desc = MathTex(
            r"N\to\infty\ \Rightarrow\ \mathrm{COI}\to 0",
            font_size=28,
            color=GREY_B,
        ).to_edge(DOWN)

        self.play(FadeIn(desc, shift=UP * 0.2))
        self.wait(2)

BEHAVIOR_SCENES = [
    "LogsToKernelsScene",
    "KLSeparabilityAndSignificanceScene",
    "TrajectorySamplingScene",
    "KroneckerExpansionScene",
]

COI_SCENES = [
    "SamplingAndReservationScene",
    "COIDistributionScene",
    "COIErosionMathScene",
]
