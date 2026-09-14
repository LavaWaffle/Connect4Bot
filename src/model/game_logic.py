import numpy as np
import numpy.typing as npt

NUM_ROWS = 6
NUM_COLS = 7


def get_init_board() -> npt.NDArray[np.int64]:
    return np.zeros((NUM_ROWS, NUM_COLS), dtype=int)


# Parameters:
#    Board: 6x7 numpy array. Game state
#    Player: 1 or -1. Who is playing
#    Action: [0, 6]. Which column should piece go to
#
# Does not modify original board, returns new updated board
def place_piece(
    board: npt.NDArray[np.int64], player: int, action: int
) -> npt.NDArray[np.int64]:
    board = np.copy(board)
    new_row_index: int = sum(board[:, action] == 0) - 1
    board[new_row_index, action] = player
    return board


def get_valid_moves(board: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    # Checks if first spot in col is empty
    return (board[0, :] == 0) * 1


def is_board_full(board: npt.NDArray[np.int64]) -> np.bool_:
    # Entire first row is empty
    return np.all(board[0] != 0)


# has player won
def is_win(board: npt.NDArray[np.int64], player: int):
    # Boolean matrix of current players pieces
    p = board == player

    # Horizontal check
    if np.any(p[:, :-3] & p[:, 1:-2] & p[:, 2:-1] & p[:, 3:]):
        return True

    # Vertical check
    if np.any(p[:-3, :] & p[1:-2, :] & p[2:-1, :] & p[3:, :]):
        return True

    # Diagonal check (\)
    if np.any(p[:-3, :-3] & p[1:-2, 1:-2] & p[2:-1, 2:-1] & p[3:, 3:]):
        return True

    # Diagonal check (/)
    return np.any(p[3:, :-3] & p[2:-1, 1:-2] & p[1:-2, 2:-1] & p[:-3, 3:])
