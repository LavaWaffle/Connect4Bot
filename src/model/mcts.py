from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from math import sqrt

import numpy as np
import numpy.typing as npt

from model.ui import render

from .game_logic import get_valid_moves, is_board_full, is_win, place_piece

NUM_SIMS = 100


@dataclass
class Node:
    prior: np.float32 | float
    turn: int
    state: npt.NDArray[np.int64]
    children: dict[int, Node] = field(default_factory=dict)
    value: float = 0.0
    visits: int = 0

    def expand(self, action_probs: npt.NDArray[np.float32]) -> None:
        valid_moves: npt.NDArray[np.int64] = get_valid_moves(self.state)
        masked_probs = action_probs * valid_moves

        total_prob = np.sum(masked_probs)
        if total_prob > 0:
            masked_probs /= total_prob
        else:
            masked_probs = valid_moves / np.sum(valid_moves)

        for action, valid in enumerate(valid_moves):
            if valid:
                next_state = place_piece(
                    board=self.state, player=self.turn, action=action
                )

                self.children[action] = Node(
                    prior=masked_probs[action], turn=self.turn * -1, state=next_state
                )

    def _ucb_score(self, child: Node) -> float:
        prior_score = (
            float(child.prior) * sqrt(max(1, self.visits)) / (child.visits + 1)
        )

        if child.visits == 0:
            value_score = 0.0
        else:
            value_score = child.value / child.visits * self.turn

        return value_score + prior_score

    def select_child(self) -> tuple[int, Node]:
        max_score = float("-inf")
        selected_action = 0
        for action, child in self.children.items():
            score = self._ucb_score(child)
            if score > max_score:
                max_score = score
                selected_action = action

        return selected_action, self.children[selected_action]

    def to_dict(self, max_depth: int = 2, action: int | None = None) -> dict:
        """Recursively converts the node and its children into a JSON-serializable dictionary."""
        q_val = (self.value / self.visits * self.turn) if self.visits > 0 else 0.0

        node_dict = {
            "action": action,
            "turn": int(self.turn),
            "visits": int(self.visits),
            "value": round(float(self.value), 3),
            "q_value": round(float(q_val), 3),
            "prior": round(float(self.prior), 3),
            "children": [],
        }

        if max_depth > 0:
            for act, child in self.children.items():
                node_dict["children"].append(
                    child.to_dict(max_depth=max_depth - 1, action=act)
                )

            # Sort children by visit count descending so the best moves appear first
            node_dict["children"].sort(key=lambda x: x["visits"], reverse=True)

        return node_dict


def get_action_probs(root: Node) -> npt.NDArray[np.float32]:
    action_probs = np.zeros(7, dtype=np.float32)
    for action, child in root.children.items():
        action_probs[action] = child.visits

    total_visits = np.sum(action_probs)
    if total_visits > 0:
        action_probs /= total_visits
    return action_probs


def run_batched_mcts(
    boards: list[npt.NDArray[np.int64]],
    players: list[int],
    predict_batch_fn: Callable,
    num_sims: int = NUM_SIMS,
    return_roots: bool = False,
    add_noise: bool = False,
    noise_epsilon: float = 0.25,
    noise_alpha: float = 0.8,
) -> list[npt.NDArray[np.float32]] | tuple[list[npt.NDArray[np.float32]], list[Node]]:
    roots = [Node(prior=0.0, turn=p, state=b) for b, p in zip(boards, players)]

    _, initial_policies = predict_batch_fn(boards, players)
    for root, policy in zip(roots, initial_policies):
        root.expand(policy)

    if add_noise:
        for root in roots:
            actions = list(root.children.keys())
            n = len(actions)
            if n > 0:
                noise = np.random.dirichlet([noise_alpha] * n)
                for action, eta in zip(actions, noise):
                    child = root.children[action]
                    child.prior = (
                        1 - noise_epsilon
                    ) * child.prior + noise_epsilon * eta

    for _ in range(num_sims):
        search_paths = []
        leaf_nodes = []

        states_to_eval = []
        turns_to_eval = []
        eval_indicies = []

        leaf_values = [0.0] * len(roots)

        # Selection
        for i, root in enumerate(roots):
            node = root
            path = [node]

            # Find next unexpanded node
            while node.children:
                _, node = node.select_child()
                path.append(node)

            search_paths.append(path)
            leaf_nodes.append(node)

            # Check end states
            if is_board_full(node.state):
                leaf_values[i] = 0.0
            elif is_win(node.state, player=1):
                leaf_values[i] = 1.0
            elif is_win(node.state, player=-1):
                leaf_values[i] = -1.0
            else:  # Game not over, queue for batch eval
                states_to_eval.append(node.state)
                turns_to_eval.append(node.turn)
                eval_indicies.append(i)

        # Batch eval and expansion
        if states_to_eval:
            values, policies = predict_batch_fn(states_to_eval, turns_to_eval)
            for idx, v, p in zip(eval_indicies, values, policies):
                leaf_values[idx] = v
                leaf_nodes[idx].expand(p)

        # Backprop
        for path, value in zip(search_paths, leaf_values):
            for node in path:
                node.value += value
                node.visits += 1

    policies = [get_action_probs(root) for root in roots]
    if return_roots:
        return policies, roots
    return policies


def run_mcts(
    board: npt.NDArray[np.int64],
    player: int,
    predict_fn: Callable[
        [npt.NDArray[np.int64], int], tuple[float, npt.NDArray[np.float32]]
    ],
    num_sims: int = NUM_SIMS,
) -> npt.NDArray[np.float32]:
    root = Node(prior=0.0, turn=player, state=board)

    value, action_probs = predict_fn(root.state, root.turn)
    root.expand(action_probs)

    for _ in range(num_sims):
        node = root
        search_path = [node]

        # find next unexpanded node
        while node.children:
            _, node = node.select_child()
            search_path.append(node)

        # check end states
        if is_board_full(node.state):
            value = 0.0
        elif is_win(node.state, player=1):
            value = 1.0
        elif is_win(node.state, player=-1):
            value = -1.0
        else:  # game not over, expand node
            value, action_probs = predict_fn(node.state, node.turn)
            node.expand(action_probs)

        # backprop value and visits along search path
        for node in search_path:
            node.value += value
            node.visits += 1

    return get_action_probs(root)


if __name__ == "__main__":
    board = np.array(
        [
            [0, -1, -1, -1, 1, 0, -1],
            [0, 1, -1, 1, 1, 0, 1],
            [-1, 1, -1, 1, 1, 0, -1],
            [1, -1, 1, -1, -1, 0, -1],
            [-1, -1, 1, -1, 1, 1, -1],
            [-1, 1, 1, -1, 1, -1, 1],
        ]
    )

    render(board)

    def dummy_predict_fn(
        state: npt.NDArray[np.int64], turn: int
    ) -> tuple[float, npt.NDArray[np.float32]]:
        value = 0.5
        policy = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0], dtype=np.float32)
        return value, policy

    print("Running MCTS simulation...")
    final_action_probs = run_mcts(
        board=board, player=1, predict_fn=dummy_predict_fn, num_sims=NUM_SIMS
    )

    print("\nResulting Action Probabilities:")
    for col, prob in enumerate(final_action_probs):
        print(f"Column {col}: {prob:.3f}")
