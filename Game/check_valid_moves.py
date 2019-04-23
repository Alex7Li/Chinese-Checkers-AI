from collections import deque
import numpy as np
from Data.grids import rotate_move_to_player


def spot_type(grid, x, y):
    """
    return -1 if this spot is invalid
    return 0 if this spot is unoccupied
    return 1 if this spot is occupied
    """
    if x < 0 or y < 0 or x >= len(grid) or y >= len(grid) or grid[x][y] == 9:
        return -1
    if grid[x][y] == 0:
        return 0
    return 1


def is_in_valid_range(cur_player, x, y):
    """
    Make sure that the player isn't going to a place that
    will confused the CPU. Let's keep this battle out of the
    outskirts :P
    :param cur_player: The player u[ to move
    :param x: The x-index of the piece to move
    :param y: The y-index of the piece
    :return: If a move is in the valid range
    """
    rotx, roty = rotate_move_to_player(3, cur_player, x, y)
    return 3 <= rotx < 10 and 3 <= roty < 10


def all_valid(grid, stX, stY):
    """
    :param grid: The current board state
    :param stX: The x-index of the piece to move
    :param stY: The y-index of the piece to move
    :return: A matrix of all valid moves
    """
    valid = np.zeros((len(grid), len(grid)), np.int8)
    if grid[stX][stY] == 9 or grid[stX][stY] == 0:
        return valid
    pos = deque()
    pos.append((stX, stY))
    valid[stX][stY] = True
    for nx, ny in [(stX + 1, stY + 1), (stX - 1, stY - 1), (stX + 1, stY),
                   (stX - 1, stY), (stX, stY + 1), (stX, stY - 1)]:
        if spot_type(grid, nx, ny) == 0:
            # note we can never jump to these squares by a chain of
            # jumps so we will never add these positions to the queue
            valid[nx][ny] = True
    while pos:
        x, y = pos.pop()
        for nx, ny, jx, jy in [(x + 1, y + 1, x + 2, y + 2),
                               (x - 1, y - 1, x - 2, y - 2),
                               (x + 1, y, x + 2, y),
                               (x - 1, y, x - 2, y),
                               (x, y + 1, x, y + 2),
                               (x, y - 1, x, y - 2)]:
            if spot_type(grid, jx, jy) == 0 and not valid[jx][jy] and \
                    spot_type(grid, nx, ny) == 1:
                pos.append((jx, jy))
                valid[jx][jy] = True
    valid[stX][stY] = False
    return valid


def is_valid(grid, stX, stY, endX, endY):
    """
    :param grid: The current board state
    :param stX: The x-index of the piece to move
    :param stY: The y-index of the piece to move
    :param endX: The x-index to move the piece to
    :param endY: The y-index to move the piece to
    :return: If you can move a piece from (stX, stY) to (endX, endY)
    """
    # if stX == endX and stY==endY:
    #    return True
    if grid[endX][endY] != 0:
        return False
    validGrid = all_valid(grid, stX, stY)
    return validGrid[endX][endY]
