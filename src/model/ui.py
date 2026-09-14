import sys

import numpy as np
import numpy.typing as npt
import pygame
from pygame.surface import Surface

COLUMN_COUNT = 7
ROW_COUNT = 6
SQUARESIZE = 100
RADIUS = int(SQUARESIZE / 2 - 5)

BLUE = (52, 186, 235)
GREY = (70, 71, 70)
WHITE = (255, 255, 255)
YELLOW = (230, 230, 20)


def draw_board(screen: Surface, board: npt.NDArray[np.int64]):
    width = COLUMN_COUNT * SQUARESIZE
    height = (ROW_COUNT + 1) * SQUARESIZE

    size = (width, height)
    board = np.flip(board, 0)

    # Draw connect 4 background
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            _ = pygame.draw.rect(
                screen,
                GREY,
                (c * SQUARESIZE, r * SQUARESIZE + SQUARESIZE, SQUARESIZE, SQUARESIZE),
            )
            _ = pygame.draw.circle(
                screen,
                WHITE,
                (
                    int(c * SQUARESIZE + SQUARESIZE / 2),
                    int(r * SQUARESIZE + SQUARESIZE + SQUARESIZE / 2),
                ),
                RADIUS,
            )

    # Draw pieces
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            if board[r][c] == 1:
                _ = pygame.draw.circle(
                    screen,
                    YELLOW,
                    (
                        int(c * SQUARESIZE + SQUARESIZE / 2),
                        height - int(r * SQUARESIZE + SQUARESIZE / 2),
                    ),
                    RADIUS,
                )
            elif board[r][c] == -1:
                _ = pygame.draw.circle(
                    screen,
                    BLUE,
                    (
                        int(c * SQUARESIZE + SQUARESIZE / 2),
                        height - int(r * SQUARESIZE + SQUARESIZE / 2),
                    ),
                    RADIUS,
                )
    pygame.display.update()


def draw_policies(screen: Surface, policies: npt.NDArray[np.float32]):
    # The total window height based on your existing UI logic
    height = (ROW_COUNT + 1) * SQUARESIZE

    for c in range(COLUMN_COUNT):
        # Width is the policy float multiplied by the column width (SQUARESIZE)
        line_width = int(policies[c] * SQUARESIZE)

        # Draw the line at the bottom of the screen (height - 2) with a height of 2
        _ = pygame.draw.rect(
            screen,
            (255, 0, 0),  # Red for high visibility against the white/grey board
            (c * SQUARESIZE, height - 2, line_width, 2),
        )

    pygame.display.update()


def render(board: npt.NDArray[np.int64]):
    _ = pygame.init()
    screen: Surface = pygame.display.set_mode((700, 700))

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            draw_board(screen, board)
            pygame.display.update()
