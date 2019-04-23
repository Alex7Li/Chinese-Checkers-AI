import numpy as np


def get_start_grid(size=3):
    """
    :param size: size of the grid (always 3, 4 functionality removed)
    :return: the 13 by 13 chinese checkers grid
    """
    st_grid = np.array([[9, 9, 9, 2, 9, 9, 9, 9, 9, 9, 9, 9, 9],
                        [9, 9, 9, 2, 2, 9, 9, 9, 9, 9, 9, 9, 9],
                        [9, 9, 9, 2, 2, 2, 9, 9, 9, 9, 9, 9, 9],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 9, 9],
                        [9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 9, 9],
                        [9, 9, 0, 0, 0, 0, 0, 0, 0, 0, 9, 9, 9],
                        [9, 9, 9, 0, 0, 0, 0, 0, 0, 0, 9, 9, 9],
                        [9, 9, 9, 3, 0, 0, 0, 0, 0, 0, 1, 9, 9],
                        [9, 9, 9, 3, 3, 0, 0, 0, 0, 0, 1, 1, 9],
                        [9, 9, 9, 3, 3, 3, 0, 0, 0, 0, 1, 1, 1],
                        [9, 9, 9, 9, 9, 9, 9, 0, 0, 0, 9, 9, 9],
                        [9, 9, 9, 9, 9, 9, 9, 9, 0, 0, 9, 9, 9],
                        [9, 9, 9, 9, 9, 9, 9, 9, 9, 0, 9, 9, 9]])
    return st_grid


def get_border_grid():
    border_grid = np.array([[9, 9, 9, 0, 9, 9, 9, 9, 9, 9, 9, 9, 9],
                            [9, 9, 9, 0, 0, 9, 9, 9, 9, 9, 9, 9, 9],
                            [9, 9, 9, 0, 0, 0, 9, 9, 9, 9, 9, 9, 9],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 9, 9],
                            [9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 9, 9],
                            [9, 9, 0, 0, 0, 0, 0, 0, 0, 0, 9, 9, 9],
                            [9, 9, 9, 0, 0, 0, 0, 0, 0, 0, 9, 9, 9],
                            [9, 9, 9, 0, 0, 0, 0, 0, 0, 0, 0, 9, 9],
                            [9, 9, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9],
                            [9, 9, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [9, 9, 9, 9, 9, 9, 9, 0, 0, 0, 9, 9, 9],
                            [9, 9, 9, 9, 9, 9, 9, 9, 0, 0, 9, 9, 9],
                            [9, 9, 9, 9, 9, 9, 9, 9, 9, 0, 9, 9, 9]])
    return border_grid


def rotate_left(grid):
    """
    Rotate the given grid 120 degrees counterclockwise.
    :param grid: grid to rotate
    """
    for i in range(0, 6):
        for j in range(0, 7):
            if grid[i][j] != 9:
                temp = grid[i][j]
                grid[i][j] = grid[-i + j + 6][-i + 12]
                i, j = - i + j + 6, -i + 12
                grid[i][j] = grid[-i + j + 6][-i + 12]
                grid[-i + j + 6][-i + 12] = temp
                i, j = 12 - j, i - j + 6


def rotate_right(grid):
    """
    Rotate the given grid 120 degrees clockwise.
    :param grid: grid to rotate
    """
    for i in range(0, 6):
        for j in range(0, 7):
            if grid[i][j] != 9:
                temp = grid[i][j]
                grid[i][j] = grid[12 - j][i - j + 6]
                i, j = 12 - j, i - j + 6
                grid[i][j] = grid[12 - j][i - j + 6]
                grid[12 - j][i - j + 6] = temp
                i, j = - i + j + 6, -i + 12


def rotate_to_player(grid, cur_player, new_player):
    """
    If we are currently looking from the perspective of cur_player,
    this function will allow us to see from the perspective of new_player
    :param grid: the current game state
    :param cur_player: the player whose view you were seeing
    :param new_player: the player whose view you want to see
    """
    if cur_player % 3 + 1 == new_player:
        rotate_right(grid)
    elif cur_player == new_player % 3 + 1:
        rotate_left(grid)


def rotate_move_to_player(cur_player, new_player, oldi, oldj):
    """
    Given a move from the perspective of cur_player,
    this function will allow us to see from the perspective of new_player
    :param grid: the current game state
    :param cur_player: the player whose view you were seeing
    :param new_player: the player whose view you want to see
    :param oldi: i-index of the move from the current players perspective
    :param oldj: j-index of the move from the current players perspective
    :return:
    """
    if cur_player % 3 + 1 == new_player:
        return -oldi + oldj + 6, -oldi + 12
    elif cur_player == new_player % 3 + 1:
        return 12 - oldj, oldi - oldj + 6
    else:
        assert cur_player == new_player
        return oldi, oldj
