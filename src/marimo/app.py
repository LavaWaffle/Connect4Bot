import marimo
from model import mcts

__generated_with = "0.24.2"
app = marimo.App(app_title="Connect 4 AI - Anand Parab")


@app.cell
def setup_imports():
    import math
    import os
    import sys

    import altair as alt
    import anywidget
    import numpy as np
    import pandas as pd
    import torch
    import traitlets

    import marimo as mo

    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.abspath(os.path.join(current_dir, ".."))
    if src_dir not in sys.path:
        sys.path.append(src_dir)

    from model.game_logic import (
        get_init_board,
        get_valid_moves,
        is_board_full,
        is_win,
        place_piece,
    )
    from model.mcts import run_batched_mcts
    from model.model import Connect4Model
    from model.train import predict_batch_fn

    return (
        Connect4Model,
        anywidget,
        current_dir,
        get_init_board,
        get_valid_moves,
        is_board_full,
        is_win,
        math,
        mo,
        alt,
        np,
        os,
        pd,
        place_piece,
        predict_batch_fn,
        run_batched_mcts,
        src_dir,
        torch,
        traitlets,
    )


@app.cell
def create_widget_class(anywidget, current_dir, os, traitlets):
    # Board Widget
    esm_path = os.path.join(current_dir, "board", "src", "board", "static", "index.js")
    if not os.path.exists(esm_path):
        esm_path = os.path.join(
            current_dir, "board", "src", "board", "static", "widget.js"
        )

    class Connect4Widget(anywidget.AnyWidget):
        _esm = esm_path

        # Synchronized states
        board = traitlets.List().tag(sync=True)
        frozen = traitlets.Bool(False).tag(sync=True)
        scale = traitlets.Float(1.0).tag(sync=True)
        done = traitlets.Bool(False).tag(sync=True)
        policy_raw = traitlets.List([]).tag(sync=True)
        policy_temp = traitlets.List([]).tag(sync=True)
        disable_animations = traitlets.Bool(False).tag(sync=True)

    # Tree Widget
    tree_esm_path = os.path.join(
        current_dir, "tree", "src", "tree", "static", "index.js"
    )
    if not os.path.exists(tree_esm_path):
        tree_esm_path = os.path.join(
            current_dir, "tree", "src", "tree", "static", "widget.js"
        )

    class MCTSTreeWidget(anywidget.AnyWidget):
        _esm = tree_esm_path
        tree_data = traitlets.Dict({}).tag(sync=True)

    return Connect4Widget, MCTSTreeWidget


@app.cell
def define_ai_params():
    ai_params = {"num_sims": 100, "temperature": 0.2}
    return (ai_params,)


@app.cell
def init_game(
    Connect4Model,
    Connect4Widget,
    MCTSTreeWidget,
    ai_params,
    get_init_board,
    get_valid_moves,
    is_board_full,
    is_win,
    math,
    np,
    os,
    place_piece,
    predict_batch_fn,
    run_batched_mcts,
    src_dir,
    torch,
):
    # Setup Device
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # Load Model Weights
    model = Connect4Model(device)
    checkpoint_path = os.path.join(src_dir, "..", "checkpoints", "best_model.pth")
    if os.path.exists(checkpoint_path):
        model.load_state_dict(
            torch.load(checkpoint_path, map_location=device, weights_only=True)
        )
    else:
        print(
            f"Warning: No weights found at {checkpoint_path}. Using random initialization."
        )
    model.eval()

    def predict_fn(states, turns):
        return predict_batch_fn(states, turns, model, device)

    # Warm up AI
    print("Warming up AI...")
    predict_fn([get_init_board()], [1])

    # Encapsulate state so the observer can mutate it safely
    class GameState:
        def __init__(self):
            self.board = get_init_board()
            self.game_over = False
            self.is_processing = False
            self.mcts_root_board = get_init_board()

    state = GameState()
    widget = Connect4Widget(board=state.board.tolist(), done=state.game_over)
    tree_widget = MCTSTreeWidget(tree_data={})

    def handle_tree_msg(tree_w, msg, buffers):
        msg_type = msg.get("type")

        if msg_type == "tree_node_clicked":
            action = msg.get("action")
            path = msg.get("path", [])

            # Use the existing state.board reference
            preview_board = np.copy(state.mcts_root_board)
            current_player = -1

            for move in path:
                if move is not None:
                    preview_board = place_piece(
                        preview_board, player=current_player, action=move
                    )
                    current_player *= -1

            widget.disable_animations = True
            widget.board = preview_board.tolist()

        elif msg_type == "tree_mouse_leave":
            # Revert to the actual game state
            widget.disable_animations = False
            widget.board = state.board.tolist()
            print("Reset")

    tree_widget.on_msg(handle_tree_msg)

    def handle_msg(widget, msg, buffers):
        msg_type = msg.get("type")

        # --- Handle Game Restart ---
        if msg_type == "restart":
            state.board = get_init_board()
            state.mcts_root_board = get_init_board()
            state.game_over = False
            widget.board = state.board.tolist()
            widget.done = False
            tree_widget.tree_data = {}
            print("Game Restarted!")
            return

        # --- Handle Column Click ---
        if msg_type == "click":
            col = msg.get("col")

            if state.game_over or state.is_processing:
                return

            valid_moves = get_valid_moves(state.board)
            if not valid_moves[col]:
                return

            state.is_processing = True

            try:
                # --- 1. Human Turn ---
                state.board = place_piece(state.board, player=1, action=col)
                widget.board = state.board.tolist()

                if is_win(state.board, 1):
                    print("Human Wins!")
                    state.game_over = True
                    widget.done = True
                    return
                elif is_board_full(state.board):
                    print("Draw!")
                    state.game_over = True
                    widget.done = True
                    return

                # --- 2. AI Turn ---
                policies, roots = run_batched_mcts(
                    [state.board],
                    [-1],
                    predict_fn,
                    num_sims=ai_params["num_sims"],
                    return_roots=True,
                )
                policy = policies[0]
                ai_root = roots[0]

                tree_data = ai_root.to_dict(max_depth=2)

                def keep_top_three_children(node, root_policy=None):
                    children = node.get("children", [])
                    is_root = root_policy is not None

                    if is_root:
                        # The root uses the final MCTS policy based on visit counts.
                        children = sorted(
                            children,
                            key=lambda child: float(root_policy[child["action"]]),
                            reverse=True,
                        )
                    else:
                        # Descendants use their per-node model policy (`prior`).
                        children = sorted(
                            children,
                            key=lambda child: float(child["prior"]),
                            reverse=True,
                        )

                    # Truncate to top 3
                    top_children = children[:3]

                    if is_root:
                        # Reorder visually: 1st in middle, 2nd above, 3rd below
                        reordered = []
                        for i, child in enumerate(top_children):
                            if i % 2 == 1:
                                # Insert at the front (renders above)
                                reordered.insert(0, child)
                            else:
                                # Append to the back (renders below)
                                reordered.append(child)
                        node["children"] = reordered
                    else:
                        node["children"] = top_children

                    for child in node["children"]:
                        # Descendants do not receive the root_policy, keeping their default sorting
                        keep_top_three_children(child)

                keep_top_three_children(tree_data, root_policy=policy)
                tree_widget.tree_data = tree_data

                temperature = ai_params["temperature"]
                if math.isclose(temperature, 0.0):
                    adjusted_policy = np.zeros_like(policy)
                    adjusted_policy[np.argmax(policy)] = 1.0
                else:
                    adjusted_policy = np.power(policy, 1.0 / temperature)
                    adjusted_policy /= np.sum(adjusted_policy)
                widget.policy_raw = policy.tolist()
                widget.policy_temp = adjusted_policy.tolist()

                state.mcts_root_board = np.copy(state.board)

                best_col = int(np.random.choice(7, p=adjusted_policy))
                state.board = place_piece(state.board, player=-1, action=best_col)
                widget.board = state.board.tolist()

                if is_win(state.board, -1):
                    print("AI Wins!")
                    state.game_over = True
                    widget.done = True
                elif is_board_full(state.board):
                    print("Draw!")
                    state.game_over = True
                    widget.done = True

            finally:
                state.is_processing = False
                widget.send({"type": "turn_complete"})

    widget.on_msg(handle_msg)
    return tree_widget, widget


@app.cell
def dynamic_sliders(ai_params, math, mo, np, widget):
    sim_slider = mo.ui.slider(
        start=50,
        stop=500,
        step=25,
        value=100,
        label="Number of Monte Carlo Simulations",
        show_value=True,
        on_change=lambda v, p=ai_params: ai_params.update({"num_sims": v}),
    )

    def on_temp_change(v):
        # Update the parameter state
        ai_params["temperature"] = v

        # Recalculate if we have a valid policy_raw
        raw = widget.policy_raw
        if raw and len(raw) == 7:
            policy = np.array(raw)
            if math.isclose(v, 0.0):
                adjusted_policy = np.zeros_like(policy)
                adjusted_policy[np.argmax(policy)] = 1.0
            else:
                adjusted_policy = np.power(policy, 1.0 / v)
                policy_sum = np.sum(adjusted_policy)
                if policy_sum > 0:
                    adjusted_policy /= policy_sum

            widget.policy_temp = adjusted_policy.tolist()

    temp_slider = mo.ui.slider(
        start=0.0,
        stop=2.0,
        step=0.1,
        value=0.2,
        label="Policy Temperature",
        show_value=True,
        on_change=on_temp_change,
    )

    # Assign to a variable and return as a tuple
    sliders_view = mo.hstack([temp_slider, sim_slider])
    return (sliders_view,)


@app.cell
def explain_project_ui(mo):
    project_explanation = mo.md(
        """
        # Connect 4 AI - Anand Parab
        Welcome to the Connect 4 AI simulation. This project features an AI agent
            trained entirely through self-play using Monte Carlo Tree Search (MCTS) and a
            residual neural network.
        """
    )
    project_explanation
    return


@app.cell
def display_game_ui(mo, sliders_view, tree_widget, widget):
    # Easy layout controls: adjust the gap here to change the spacing between panels.
    play_game = mo.md(
        """
        ## Project Demo: Play against the Connect 4 AI
        """
    )
    board_panel = mo.vstack(
        [mo.md("### Board"), widget],
        align="center",
        gap=-5,
    )
    tree_panel = mo.vstack(
        [mo.md("### MCTS Tree"), tree_widget],
        align="center",
        gap=0,
    )

    board_and_tree = mo.hstack(
        [board_panel, tree_panel],
        justify="start",
        align="start",
        gap=0,
    )
    game_interface = mo.vstack([play_game, sliders_view, board_and_tree])

    game_interface
    return


@app.cell
def game_vals_explained(mo):
    mo.md(
        """
        <style>
            /* Remove bottom margin from the bolded section titles (paragraphs) */
            p {
                margin-bottom: 0.2em !important;
            }
            /* Remove top margin from the lists so they sit flush against the text */
            ul, ol {
                margin-top: -0.5em !important;
                margin-bottom: -0.5em !important;
                padding-top: 0 !important;
                padding-bottom: 0 !important;
            }
            /* Optional: tighten the spacing between the bullet points themselves */
            li {
                margin-bottom: 0.0em !important;
            }
        </style>
        ### Game Values Explained
        1. **Q-Value**: Score between -1 (loss) and 1 (win) estimating the current player's change of winning (from their alternating viewpoint at each level of the tree).
        2. **Visits**: The number of times the Monte Carlo Tree Search visited a specific state.
        3. **Prior**: The initial probability assigned to a move by the neural network's policy head.
        """
    )


@app.cell
def metrics_section(mo):
    metrics_explanation = mo.md(
        """
        ## Training Metrics & Analysis
        """
    )
    metrics_explanation


@app.cell
def train_df_cell(current_dir, os, pd):
    train_metrics_path = os.path.join(current_dir, "csvs", "train_metrics.csv")
    train_df = pd.read_csv(train_metrics_path)
    return (train_df,)


@app.cell
def loss_chart_cell(mo, train_df, alt):

    # ==========================================
    # CONFIGURATION CONSTANTS
    # ==========================================
    CHART_WIDTH = 500
    CHART_HEIGHT = 350
    CHART_TITLE = "Training Metrics"
    X_AXIS_TITLE = "Epoch"
    Y_AXIS_TITLE = ""

    # Data columns to plot
    METRICS_TO_PLOT = ["epoch", "loss", "value_loss", "policy_loss", "entropy"]

    # Acceptance Mark Settings
    SHOW_ACCEPTANCE_MARKS = False
    MARK_COLOR = "red"
    MARK_SIZE = 15
    MARK_THICKNESS = 2
    # ==========================================

    # 1. Melt the dataframe using the constants
    loss_df = train_df[METRICS_TO_PLOT].melt(
        "epoch", var_name="metric", value_name="value"
    )

    # 2. Filter for only the epochs where the model was accepted
    accepted_df = train_df[train_df["accepted"] == 1]

    # 3. Tooltip selection
    nearest = alt.selection_point(
        nearest=True, on="mouseover", fields=["epoch"], empty=False
    )

    # 4. The base lines
    line = (
        alt.Chart(loss_df)
        .mark_line()
        .encode(
            x=alt.X("epoch", title=X_AXIS_TITLE),
            y=alt.Y("value", title=Y_AXIS_TITLE),
            color="metric",
        )
    )

    # 5. The tooltip selectors
    selectors = (
        alt.Chart(loss_df)
        .mark_point()
        .encode(
            x="epoch",
            y="value",
            opacity=alt.value(0),
            tooltip=["epoch", "metric", "value"],
        )
        .add_params(nearest)
    )

    # 6. The dynamic points that appear on hover
    points = line.mark_point().encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )

    # 7. Add layers dynamically based on configurations
    layers = [line, selectors, points]

    if SHOW_ACCEPTANCE_MARKS:
        acceptance_marks = (
            alt.Chart(accepted_df)
            .mark_tick(color=MARK_COLOR, size=MARK_SIZE, thickness=MARK_THICKNESS)
            .encode(
                x="epoch",
                y=alt.value(CHART_HEIGHT),  # Locks the mark to the dynamic chart height
                tooltip=["epoch", "win_rate_pct"],
            )
        )
        layers.append(acceptance_marks)

    # 8. Layer them all together using the unpacked list
    loss_chart = alt.layer(*layers).properties(
        title=CHART_TITLE, width=CHART_WIDTH, height=CHART_HEIGHT
    )

    loss_chart_key = mo.md(
        r"""
            <style>
                /* Remove bottom margin from the bolded section titles (paragraphs) */
                p {
                    margin-bottom: 0.2em !important;
                }
                /* Remove top margin from the lists so they sit flush against the text */
                ul, ol {
                    margin-top: -0.5em !important;
                    margin-bottom: -0.5em !important;
                    padding-top: 0 !important;
                    padding-bottom: 0 !important;
                }
                /* Optional: tighten the spacing between the bullet points themselves */
                li {
                    margin-bottom: 0.0em !important;
                }
            </style>

            * **Loss (`loss`):** The overall error the neural network is trying to minimize during training. $L = L_{value} + L_{policy}$
            * **Policy Loss (`policy_loss`):** Measures how accurately the network's raw move predictions match the high-quality target policy generated by Monte Carlo Tree Search (MCTS). $L_{policy} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{a} p_{a,i} \log(\hat{p}_{a,i})$ where $N$ is the batch size (the number of board positions processed in a single training step).
            * **Value Loss (`value_loss`):** Measures how well the network predicts the final outcome of the game from a specific board state. $L_{value} = \frac{1}{N} \sum_{i=1}^{N} (v_i - \hat{v}_i)^2$ where $N$ is the batch size. (1 for a win, -1 for a loss, 0 for a draw).
            * **Entropy:** A measure of uncertainty or randomness in the MCTS policy's move selection during self-play. It is calculated using Shannon entropy: $H = -\sum p_i \log(p_i)$ where $p_i$ is the probability of selecting a specific move. High entropy indicates the model is unsure and exploring many options, while low entropy means the model is highly confident in a specific move.
        """
    )

    loss_chart_desc = mo.md(
        """
        <style>
            /* Remove bottom margin from the bolded section titles (paragraphs) */
            p {
                margin-bottom: 0.2em !important;
            }
            /* Remove top margin from the lists so they sit flush against the text */
            ul, ol {
                margin-top: -0.5em !important;
                margin-bottom: -0.5em !important;
                padding-top: 0 !important;
                padding-bottom: 0 !important;
            }
            /* Optional: tighten the spacing between the bullet points themselves */
            li {
                margin-bottom: 0.0em !important;
            }
        </style>
        **Learning Rate Analysis**
        * **Steep Loss Reduction:** The model demonstrates rapid initial learning, indicated by sharp declines across total loss, policy loss, and value loss in the first 10 to 20 epochs.
        * **Confidence Calibration:** Average policy entropy drops from 1.5 to ~0.8 in the first 20 epochs. This confirms the network is successfully moving away from uniform random guessing to confident move preferences.
        * **Diminishing Returns:** The training curves flatten significantly after epoch 50, with total loss bottoming out near epoch 68.

        **Future Strategies for Improvement**
        1. **More Games Per Epoch:** Increasing the volume of self-play games (currently 160 games per epoch, generated via 10 episodes of 16 batches) will expose the model to a wider distribution of board states, allowing it to explore more strategies.
        1. **Increase MCTS Simulations:** Self-play currently utilizes 200 Monte Carlo Tree Search (MCTS) simulations, while Arena evaluation uses 50. Scaling up the simulations (e.g., to 400-800) during self-play will generate significantly higher-quality target policies and value estimations.
        """
    )

    mo.vstack([loss_chart_key, loss_chart, loss_chart_desc])
    return


@app.cell
def model_arch_display(mo):
    # graph LR makes the flow horizontal (Left to Right)
    # <br> is used to add the tensor shapes on a new line inside each block
    arch_graph = mo.mermaid(
        """
        graph LR
            classDef default fill:#f9f9f9,stroke:#333,stroke-width:1px;
            classDef input fill:#e1f5fe,stroke:#0288d1;
            classDef output fill:#e8f5e9,stroke:#388e3c;
            classDef block fill:#fff3e0,stroke:#f57c00;

            Input["Input Board<br>Shape: (3, 6, 7)"]:::input --> Init["Initial Convolutional Block<br>Shape: (128, 6, 7)"]:::block

            Init --> Res1["Residual Block 1<br>Shape: (128, 6, 7)"]:::block
            Res1 --> Res2["Residual Block 2<br>Shape: (128, 6, 7)"]:::block

            Res2 --> ValueHead["Value Head<br>Shape: (1)"]:::block
            ValueHead --> OutV["Value Output<br>(-1 to 1)"]:::output

            Res2 --> PolicyHead["Policy Head<br>Shape: (7)"]:::block
            PolicyHead --> OutP["Policy Output<br>(7 probabilities)"]:::output
        """
    )

    arch_layout = mo.vstack(
        [
            mo.md("## Model Architecture"),
            mo.md(
                """
                Simplified high-level view of the Connect 4 neural network. *(Note: Batch dimension is omitted for clarity)*<br>
                Input Shape is 3 6x7 boards.<br>
                The first and second boards are one hot encoded boards showing where player 1 and -1 pieces are placed.<br>
                The third is 6x7 grid of 1s or -1s indicating the current player.<br>
                """
            ),
            arch_graph,
        ],
        align="stretch",
    )

    arch_layout
    # return (arch_layout, arch_graph)


@app.cell
def display_frozen_board(Connect4Widget, np, mo, os, current_dir):
    best_heatmap = os.path.join(current_dir, "imgs", "best_heatmap.png")
    board_array = np.array(
        [
            [0, -1, -1, -1, 1, 0, -1],
            [0, 1, -1, 1, 1, 0, 1],
            [-1, 1, -1, 1, 1, 0, -1],
            [1, -1, 1, -1, -1, 0, -1],
            [-1, -1, 1, -1, 1, 1, -1],
            [-1, 1, 1, -1, 1, -1, 1],
        ]
    )
    board_array = board_array * -1

    frozen_widget = Connect4Widget(board=board_array.tolist(), frozen=True, scale=0.55)

    mo.vstack(
        [
            mo.md(
                """
                ## AI Strategy Visualization

                The AI (blue) must play in column 6 to prevent yellow from completing a horizontal four-in-a-row on row 3 (from the bottom). <br>
                The Res 2 Rank 1 filter highlights the AI successfully isolating column 6 as yellow's winning condition. <br>
                The Policy Head Rank 2 filter shows the AI acting on this data, concluding it must place a piece in column 6, row 3. <br>
                This demonstrates that the AI has learned to detect winning patterns and place pieces accordingly.
                """
            ),
            mo.hstack(
                [
                    frozen_widget,
                    mo.image(
                        src=best_heatmap,
                        height="375px",
                        caption="AI's Heat Map. Showing Top 2 Amplitude Filters for Relevant Layers",
                    ),
                ],
                align="center",
            ),
        ]
    )


@app.cell
def mcts_bench_df_cell(current_dir, os, pd):
    mcts_bench_df = os.path.join(current_dir, "csvs", "mcts_benchmark.csv")
    mcts_bench_df = pd.read_csv(mcts_bench_df)
    return (mcts_bench_df,)


@app.cell
def mcts_bench_chart_cell(pd, mo, mcts_bench_df, alt):
    mcts_bench_chart = (
        alt.Chart(mcts_bench_df)
        .mark_boxplot(size=40, outliers=False)  # Hides outlier dots
        .encode(
            x=alt.X("batch_size:O", title="Batch Size"),
            y=alt.Y("throughput_boards_per_sec:Q", title="Throughput (boards/sec)"),
            tooltip=alt.value(None),
        )
        .properties(title="MCTS Throughput vs Batch Size", width=500, height=350)
    )

    mo.vstack(
        [
            mo.md(
                """
                ## Monte Carlo Tree Search (MCTS) Batch Size Benchmark
                The training of the AI requires millions of runs through the MCTS algorithm.<br>
                Thus, the algorithm was designed to take a batch size parameter to make effective use of the compute resources.

                The benchmark below was performed on an Apple A18 Pro Chip. (The same chip the AI was trained on)<br>
                From batch size 1 to 64, the throughput of boards / second increased exponentially.<br>
                However, at batch size 128, the throughput plateaued as the Apple A18 Pro Chip reached its maximum throughput.<br>

                Due to the limitations of the Apple A18 Pro Chip, the training took ~8 hours for 100 epochs.<br>
                <b> In the future</b> I'd like to accelerate training using more powerful hardware to enable faster results!
                """
            ),
            mcts_bench_chart,
        ]
    )


if __name__ == "__main__":
    app.run()
