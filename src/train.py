import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset

from game_logic import (
    get_init_board,
    get_valid_moves,
    is_board_full,
    is_win,
    place_piece,
)
from mcts import run_batched_mcts
from model import Connect4Model


class Connect4Dataset(Dataset):
    def __init__(self, memory):
        self.memory = memory

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, idx: int):
        board_np, target_pi, target_v, turn = self.memory[idx]

        p1_board = (board_np == 1).astype(np.float32)
        p2_board = (board_np == -1).astype(np.float32)
        turn_board = np.full(board_np.shape, turn, dtype=np.float32)
        state = np.stack([p1_board, p2_board, turn_board])

        state_tensor = torch.tensor(state, dtype=torch.float32)
        target_pi_tensor = torch.tensor(target_pi, dtype=torch.float32)
        target_v_tensor = torch.tensor([target_v], dtype=torch.float32)

        return state_tensor, target_pi_tensor, target_v_tensor


def get_tensor_state(board: npt.NDArray[np.int64], player: int):
    # board shape (6, 7)
    p1_board = (board == 1).astype(np.float32)
    p2_board = (board == -1).astype(np.float32)
    turn_board = np.full(board.shape, player, dtype=np.float32)

    # state shape (3, 6, 7)
    state = np.stack([p1_board, p2_board, turn_board])
    # unsqueeze -> shape (1, 3, 6, 7)
    return torch.tensor(state).unsqueeze(0).to(device)


def self_play_batch_episode(num_batch: int = 10):
    boards = [get_init_board() for _ in range(num_batch)]
    players = [1] * num_batch
    active_indicies = list(range(num_batch))

    all_episode_data = {i: [] for i in range(num_batch)}
    finished_data = []

    def predict_batch_fn(states: list[npt.NDArray[np.int64]], turns: list[int]):
        batch_states = []
        for board, player in zip(states, turns):
            p1_board = (board == 1).astype(np.float32)
            p2_board = (board == -1).astype(np.float32)
            turn_board = np.full(board.shape, player, dtype=np.float32)
            batch_states.append(np.stack([p1_board, p2_board, turn_board]))

        tensor_states = torch.tensor(np.array(batch_states)).to(device)

        with torch.no_grad():
            v, p = model(tensor_states)

        p = np.exp(p.cpu().numpy())  # (Batch, 7)
        v = v.cpu().numpy().squeeze(axis=1)  # (Batch,)

        valid_moves_batch = np.array([get_valid_moves(s) for s in states])
        p *= valid_moves_batch

        row_sums = p.sum(axis=1, keepdims=True)
        p = np.divide(p, row_sums, out=np.zeros_like(p), where=row_sums != 0)

        return v, p

    while active_indicies:
        active_boards = [boards[i] for i in active_indicies]
        active_players = [players[i] for i in active_indicies]

        mcts_policies = run_batched_mcts(
            active_boards, active_players, predict_batch_fn, num_sims=100
        )

        next_active = []
        for idx, board, player, policy in zip(
            active_indicies, active_boards, active_players, mcts_policies
        ):
            all_episode_data[idx].append((board, policy, player))

            action = np.random.choice(7, p=policy)
            new_board = boards[idx] = place_piece(board, player, action)

            if is_win(new_board, player):
                finished_data.extend(
                    [
                        (b, p, float(player), turn)
                        for b, p, turn in all_episode_data[idx]
                    ]
                )
            elif is_board_full(new_board):
                finished_data.extend(
                    [(b, p, 0.0, turn) for b, p, turn in all_episode_data[idx]]
                )
            else:
                players[idx] *= -1
                next_active.append(idx)

        active_indicies = next_active
    return finished_data


def print_episode_data(episode_data):
    for step, (board, policy, value, turn) in enumerate(episode_data):
        # Map 1, -1, 0 to symbols for easy reading
        symbols = {1: "X", -1: "O", 0: "."}

        print(f"\n--- Step {step + 1} | Player {symbols[turn]} / {turn} ---")

        for row in board:
            print("  ".join(symbols[val] for val in row))

        # Format policy percentages cleanly
        formatted_policy = [f"{p * 100:02.0f}%" for p in policy]
        print(f"\nPolicy: {formatted_policy}")
        print(f"Target Value: {value}")


if __name__ == "__main__":
    print("[TRAIN] Initializing Device")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[TRAIN] Initialized Dev: {device}")

    model = Connect4Model(device)
    print("[TRAIN] Model Initialized")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print("[TRAIN] Optimizer Initialized")

    memory = []
    for i in range(5):
        print(f"[TRAIN] Self_Play_Episode: {i}")
        episode_data = self_play_batch_episode(16)
        print_episode_data(episode_data)
        memory.extend(episode_data)

    print("Finished self_play_episodes")
    model = model.train()

    dataset = Connect4Dataset(memory)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)

    for state_batch, pi_batch, v_batch in dataloader:
        # batches = truth from actual games played till end
        state_batch = state_batch.to(device)
        pi_batch = pi_batch.to(device)
        v_batch = v_batch.to(device)

        # predictions = model guesses on each state (not played till end. just one turn)
        v_pred, p_pred_log = model(state_batch)

        # (value) MSE loss + (policy) Cross Entropy Loss
        value_loss = F.mse_loss(v_pred.view(-1), v_batch.view(-1))
        policy_loss = -torch.sum(pi_batch * p_pred_log) / pi_batch.size(0)
        loss = value_loss + policy_loss

        # Backpropagate the loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
