"""Connect 4 Model and Game Logic Package."""

from model.game_logic import (
    NUM_COLS,
    NUM_ROWS,
    get_init_board,
    get_valid_moves,
    is_board_full,
    is_win,
    place_piece,
)
from model.mcts import Node, get_action_probs, run_batched_mcts, run_mcts
from model.model import Connect4Model
from model.train import Connect4Dataset, arena_eval, predict_batch_fn

__all__ = [
    # Game Logic
    "NUM_ROWS",
    "NUM_COLS",
    "get_init_board",
    "place_piece",
    "get_valid_moves",
    "is_board_full",
    "is_win",
    # Model
    "Connect4Model",
    # MCTS
    "Node",
    "get_action_probs",
    "run_mcts",
    "run_batched_mcts",
    # Training & Evaluation
    "Connect4Dataset",
    "predict_batch_fn",
    "arena_eval",
]
