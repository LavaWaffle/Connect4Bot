import sys
from time import sleep

import numpy as np
import pygame
import torch

from .game_logic import (
    get_init_board,
    get_valid_moves,
    is_board_full,
    is_win,
    place_piece,
)
from .mcts import run_batched_mcts
from .model import Connect4Model
from .train import predict_batch_fn
from .ui import SQUARESIZE, draw_board, draw_policies

TEMPERATURE = 0.2


def main():
    # Setup hardware device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Load the best model weights
    model = Connect4Model(device)
    model.load_state_dict(
        torch.load("./checkpoints/best_model.pth", map_location=device)
    )
    model.eval()

    # Warp up AI
    predict_fn = lambda states, turns: predict_batch_fn(states, turns, model, device)
    print("Initializing AI...")
    predict_fn([get_init_board()], [1])

    # Initialize Pygame and the board
    pygame.init()
    screen = pygame.display.set_mode((700, 700))
    pygame.display.set_caption("Connect 4: Play vs AI")

    board = get_init_board()
    draw_board(screen, board)

    human_player = 1
    ai_player = -1
    turn = human_player
    game_over = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_x
            ):
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                board = get_init_board()
                turn = human_player
                game_over = False
                draw_board(screen, board)
                print("Game reset.")

            # Human turn input
            if event.type == pygame.MOUSEBUTTONDOWN and turn == human_player:
                pos_x = event.pos[0]
                col = int(pos_x // SQUARESIZE)

                valid_moves = get_valid_moves(board)
                if valid_moves[col]:
                    board = place_piece(board, turn, col)
                    draw_board(screen, board)

                    if is_win(board, turn):
                        print("Human Wins!")
                        game_over = True
                    elif is_board_full(board):
                        print("Draw!")
                        game_over = True
                    else:
                        turn *= -1

        # AI Turn
        if turn == ai_player and not game_over:
            print("AI is thinking...")

            # Softmax w/ Temp
            policy = run_batched_mcts([board], [turn], predict_fn, num_sims=100)[0]
            adjusted_policy = np.power(policy, 1.0 / TEMPERATURE)
            adjusted_policy /= np.sum(adjusted_policy)

            best_col = np.random.choice(7, p=adjusted_policy)

            board = place_piece(board, turn, best_col)
            draw_board(screen, board)
            draw_policies(screen, policy)

            if is_win(board, turn):
                print("AI Wins!")
                game_over = True
            elif is_board_full(board):
                print("Draw!")
                game_over = True
            else:
                turn *= -1


if __name__ == "__main__":
    main()
