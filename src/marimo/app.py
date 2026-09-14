import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell
def setup_imports():
    import os
    import sys

    import anywidget
    import numpy as np
    import torch
    import traitlets

    # Ensure 'src' is in the system path so we can import the 'model' package
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
        np,
        os,
        place_piece,
        predict_batch_fn,
        run_batched_mcts,
        src_dir,
        torch,
        traitlets,
    )


@app.cell
def create_widget_class(anywidget, current_dir, os, traitlets):
    # Depending on your Vite config, the bundle is usually named index.js or widget.js
    esm_path = os.path.join(current_dir, "board", "src", "board", "static", "index.js")
    if not os.path.exists(esm_path):
        esm_path = os.path.join(
            current_dir, "board", "src", "board", "static", "widget.js"
        )

    class Connect4Widget(anywidget.AnyWidget):
        _esm = esm_path

        # Synchronized states
        board = traitlets.List().tag(sync=True)
        done = traitlets.Bool(False).tag(sync=True)

    return (Connect4Widget,)


@app.cell
def init_game(
    Connect4Model,
    Connect4Widget,
    get_init_board,
    get_valid_moves,
    is_board_full,
    is_win,
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

    state = GameState()
    widget = Connect4Widget(board=state.board.tolist(), done=state.game_over)

    def handle_msg(widget, msg, buffers):
        msg_type = msg.get("type")

        # --- Handle Game Restart ---
        if msg_type == "restart":
            state.board = get_init_board()
            state.game_over = False
            widget.board = state.board.tolist()
            widget.done = False
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
                policy = run_batched_mcts(
                    [state.board], [-1], predict_fn, num_sims=100
                )[0]

                temperature = 0.2
                adjusted_policy = np.power(policy, 1.0 / temperature)
                adjusted_policy /= np.sum(adjusted_policy)
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

    # Listen for custom messages instead of traitlet changes
    widget.on_msg(handle_msg)
    return (widget,)


@app.cell
def display_ui(widget):
    # Outputting the widget renders the React Component to the Marimo interface
    widget
    return


if __name__ == "__main__":
    app.run()
