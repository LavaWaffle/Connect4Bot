import copy
import os
from collections import deque

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset

from .game_logic import (
    get_init_board,
    get_valid_moves,
    is_board_full,
    is_win,
    place_piece,
)
from .mcts import run_batched_mcts
from .model import Connect4Model


def augment_with_symmetry(
    episode_data: list[tuple],
) -> list[tuple]:
    """Double training data by appending horizontally mirrored positions."""
    mirrored = [
        (np.fliplr(board).copy(), policy[::-1].copy(), value, turn)
        for board, policy, value, turn in episode_data
    ]
    return episode_data + mirrored


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


def predict_batch_fn(
    states: list[npt.NDArray[np.int64]], turns: list[int], model, device
):
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


def self_play_batch_episode(model, device, num_batch: int = 10):
    boards = [get_init_board() for _ in range(num_batch)]
    players = [1] * num_batch
    active_indicies = list(range(num_batch))

    all_episode_data = {i: [] for i in range(num_batch)}
    finished_data = []
    game_wins = 0
    game_draws = 0
    entropies: list[float] = []

    TEMP_THRESHOLD = 15  # stochastic sampling for first N moves, then argmax

    while active_indicies:
        active_boards = [boards[i] for i in active_indicies]
        active_players = [players[i] for i in active_indicies]

        mcts_policies = run_batched_mcts(
            active_boards,
            active_players,
            lambda states, turns: predict_batch_fn(states, turns, model, device),
            num_sims=200,
            add_noise=True,
        )

        next_active = []
        for idx, board, player, policy in zip(
            active_indicies, active_boards, active_players, mcts_policies
        ):
            all_episode_data[idx].append((board, policy, player))

            nonzero = policy[policy > 0]
            entropies.append(float(-np.sum(nonzero * np.log(nonzero))))

            move_number = len(all_episode_data[idx])
            if move_number < TEMP_THRESHOLD:
                action = np.random.choice(7, p=policy)
            else:
                action = int(np.argmax(policy))
            new_board = boards[idx] = place_piece(board, player, action)

            if is_win(new_board, player):
                game_wins += 1
                game_data = [
                    (b, p, float(player), turn) for b, p, turn in all_episode_data[idx]
                ]
                finished_data.extend(augment_with_symmetry(game_data))
            elif is_board_full(new_board):
                game_draws += 1
                game_data = [(b, p, 0.0, turn) for b, p, turn in all_episode_data[idx]]
                finished_data.extend(augment_with_symmetry(game_data))
            else:
                players[idx] *= -1
                next_active.append(idx)

        active_indicies = next_active

    stats = {
        "wins": game_wins,
        "draws": game_draws,
        "avg_entropy": float(np.mean(entropies)) if entropies else 0.0,
    }
    return finished_data, stats


def arena_eval(new_model, old_model, device, num_games=20, num_sims=50):
    new_model.eval()
    old_model.eval()

    boards = [get_init_board() for _ in range(num_games)]
    players = [1] * num_games
    active_indicies = list(range(num_games))

    new_model_is_p1 = np.array([(i < num_games // 2) for i in range(num_games)])

    wins_new, wins_old, draws = 0, 0, 0

    new_predict = lambda states, turns: predict_batch_fn(
        states, turns, new_model, device
    )
    old_predict = lambda states, turns: predict_batch_fn(
        states, turns, old_model, device
    )

    while active_indicies:
        new_turn_indicies, old_turn_indicies = [], []

        for idx in active_indicies:
            is_new_turn = (new_model_is_p1[idx] and players[idx] == 1) or (
                not new_model_is_p1[idx] and players[idx] == -1
            )

            if is_new_turn:
                new_turn_indicies.append(idx)
            else:
                old_turn_indicies.append(idx)

        if new_turn_indicies:
            new_boards = [boards[i] for i in new_turn_indicies]
            new_players = [players[i] for i in new_turn_indicies]
            new_policies = run_batched_mcts(
                new_boards, new_players, new_predict, num_sims
            )
        else:
            new_policies = []

        if old_turn_indicies:
            old_boards = [boards[i] for i in old_turn_indicies]
            old_players = [players[i] for i in old_turn_indicies]
            old_policies = run_batched_mcts(
                old_boards, old_players, old_predict, num_sims
            )
        else:
            old_policies = []

        next_active = []

        def process_moves(turn_indicies, policies, is_new_acting, temperature=1.0):
            nonlocal wins_new, wins_old, draws
            for i, idx in enumerate(turn_indicies):
                if temperature == 0.0:
                    action = int(np.argmax(policies[i]))
                else:
                    adjusted_policy = np.power(policies[i], 1.0 / temperature)
                    adjusted_policy /= np.sum(adjusted_policy)

                    action = int(np.random.choice(7, p=adjusted_policy))

                boards[idx] = place_piece(boards[idx], players[idx], action)

                if is_win(boards[idx], players[idx]):
                    if is_new_acting:
                        wins_new += 1
                    else:
                        wins_old += 1
                elif is_board_full(boards[idx]):
                    draws += 1
                else:
                    players[idx] *= -1
                    next_active.append(idx)

        process_moves(
            new_turn_indicies, new_policies, is_new_acting=True, temperature=0.0
        )
        process_moves(
            old_turn_indicies, old_policies, is_new_acting=False, temperature=0.0
        )

        active_indicies = next_active

    return wins_new, wins_old, draws


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

    os.makedirs("./checkpoints", exist_ok=True)

    current_model = Connect4Model(device)
    best_model = Connect4Model(device)
    best_model.load_state_dict(current_model.state_dict())

    optimizer = optim.Adam(current_model.parameters(), lr=0.001)

    NUM_EPOCHS = 100
    EPISODES_PER_EPOCH = 10
    EVAL_GAMES = 40
    WIN_THRESHOLD = 0.52
    REPLAY_BUFFER_MAX_EPOCHS = 20

    replay_buffer: deque[list] = deque(maxlen=REPLAY_BUFFER_MAX_EPOCHS)

    for epoch in range(NUM_EPOCHS):
        print(f"\n=======================")
        print(f"=== EPOCH {epoch + 1}/{NUM_EPOCHS} ===")
        print(f"=======================\n")

        epoch_memory = []
        epoch_wins = epoch_draws = 0
        epoch_entropy = 0.0
        best_model.eval()

        for i in range(EPISODES_PER_EPOCH):
            print(f"[TRAIN] Self_Play_Episode: {i + 1}/{EPISODES_PER_EPOCH}")
            episode_data, ep_stats = self_play_batch_episode(
                best_model, device, num_batch=16
            )
            epoch_memory.extend(episode_data)
            epoch_wins += ep_stats["wins"]
            epoch_draws += ep_stats["draws"]
            epoch_entropy += ep_stats["avg_entropy"]

        total_games = epoch_wins + epoch_draws
        draw_rate = epoch_draws / total_games if total_games > 0 else 0.0
        print(
            f"[TRAIN] Self-Play Stats: {total_games} games | "
            f"Draw Rate: {draw_rate * 100:.1f}% | "
            f"Avg Policy Entropy: {epoch_entropy / EPISODES_PER_EPOCH:.3f}"
        )

        replay_buffer.append(epoch_memory)
        all_memory = [sample for epoch_data in replay_buffer for sample in epoch_data]

        print(
            f"[TRAIN] Training on {len(all_memory)} positions "
            f"({len(replay_buffer)}/{REPLAY_BUFFER_MAX_EPOCHS} epochs in buffer)..."
        )
        current_model.train()
        dataset = Connect4Dataset(all_memory)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)

        total_loss = total_value_loss = total_policy_loss = 0.0
        num_batches = 0

        for state_batch, pi_batch, v_batch in dataloader:
            # batches = truth from actual games played till end
            state_batch = state_batch.to(device)
            pi_batch = pi_batch.to(device)
            v_batch = v_batch.to(device)

            # predictions = model guesses on each state (not played till end. just one turn)
            v_pred, p_pred_log = current_model(state_batch)

            # (value) MSE loss + (policy) Cross Entropy Loss
            value_loss = F.mse_loss(v_pred.view(-1), v_batch.view(-1))
            policy_loss = -torch.sum(pi_batch * p_pred_log) / pi_batch.size(0)
            loss = value_loss + policy_loss

            # Backpropagate the loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_value_loss += value_loss.item()
            total_policy_loss += policy_loss.item()
            num_batches += 1

        print(
            f"[TRAIN] Loss: {total_loss / num_batches:.4f} "
            f"(value: {total_value_loss / num_batches:.4f}, "
            f"policy: {total_policy_loss / num_batches:.4f})"
        )

        print("[TRAIN] Running Arena Evaluation...")
        wins_new, wins_old, draws = arena_eval(
            new_model=current_model,
            old_model=best_model,
            device=device,
            num_games=EVAL_GAMES,
        )

        win_rate = (wins_new + 0.5 * draws) / EVAL_GAMES
        print(f"Arena Result: New {wins_new} - {wins_old} Old (Draws: {draws})")
        print(f"Win Rate: {win_rate * 100:.1f}%")

        if win_rate > WIN_THRESHOLD:
            print("[TRAIN] Model improved! Saving new best model to ./checkpoints")
            best_model.load_state_dict(current_model.state_dict())

            torch.save(
                current_model.state_dict(), f"./checkpoints/model_epoch_{epoch + 1}.pth"
            )
            torch.save(current_model.state_dict(), "./checkpoints/best_model.pth")
        else:
            print("[TRAIN] Model failed to improve. Rejecting and reverting weights.")
            current_model.load_state_dict(best_model.state_dict())
