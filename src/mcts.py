from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt

import numpy as np
import numpy.typing as npt

from game_logic import (
    is_board_full,
    is_win,
    place_piece,
)
from ui import render

NUM_SIMS = 100


@dataclass
class Node:
    prior: np.float32 | float
    turn: int
    state: npt.NDArray[np.int64]
    children: dict[int, Node] = field(default_factory=dict)
    value: float = 0.0
    visits: int = 0

    # action_probs float for moving to each col
    def expand(self, action_probs: npt.NDArray[np.float32]) -> None:
        for action, prob in enumerate(action_probs):
            if prob > 0:
                next_state = place_piece(
                    board=self.state, player=self.turn, action=action
                )

                self.children[action] = Node(
                    prior=prob, turn=self.turn * -1, state=next_state
                )

    def _ucb_score(self, child: Node) -> float:
        prior_score = float(child.prior * sqrt(self.visits) / (child.visits + 1))

        if child.visits == 0:
            value_score = 0.0
        else:
            value_score = child.value / child.visits

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

    # initialize and expand root
    root = Node(prior=0.0, turn=1, state=board)

    def dummy_model_predict(
        board: npt.NDArray[np.int64],
    ) -> tuple[float, npt.NDArray[np.float32]]:
        value_head = 0.5
        policy_head = np.array([0.5, 0, 0, 0, 0, 0.5, 0], dtype=np.float32)
        return value_head, policy_head

    root.expand(action_probs=dummy_model_predict(root.state)[1])

    # iterate through simulation
    for _ in range(NUM_SIMS):
        node = root
        search_path = [node]

        # find next unexpanded node
        while node.children:
            action, node = node.select_child()
            search_path.append(node)

        value = None
        if is_board_full(node.state):
            value = 0.0
        if is_win(node.state, player=1):
            value = 1.0
        if is_win(node.state, player=-1):
            value = -1.0

        if value is None:
            # game not over, expand node
            value, action_probs = dummy_model_predict(node.state)
            node.expand(action_probs)

        # update value and visits along search path
        for node in search_path:
            node.value += value
            node.visits += 1

    print(root.children[0].state)
    print(root.children[0].value)

    print(root.children[5].state)
    print(root.children[5].value)
