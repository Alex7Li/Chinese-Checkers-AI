import numpy as np
from Graphics.graphics import display_victory
from Data import grids
from Game.check_valid_moves import all_valid


def update_gui_if_player_winning(screen, grid, cur_player):
    """
    check if the player has just won the game
    :param screen: The screen to update
    :param grid: The game state, oriented normally
    :param cur_player: the player to check
    :return: if this new position is winning
    """
    if is_winning(grid, cur_player):
        display_victory(screen, cur_player)
        return True
    return False


def move(grid, old_i, old_j, new_i, new_j):
    """
    grid, moving a rectangle from(oldI, oldJ) to  (newI, newJ)
    :param grid: Full board state
    :param old_i: i-index of old point
    :param old_j: j-index of old point
    :param new_i: i-index of new point
    :param new_j: j-index of new point
    """
    assert 1 <= grid[old_i][old_j] <= 3
    if old_i == new_i and old_j == new_j:
        return
    assert grid[new_i][new_j] == 0
    # assert check_valid_moves.is_valid(grid, old_i, old_j, new_i, new_j)
    grid[new_i][new_j] = grid[old_i][old_j]
    grid[old_i][old_j] = 0


def is_winning(grid, player):
    """
    Returns if a given player is currently winning
    :param grid: the 13x13 board state, oriented normally
    :param player: the player to check
    :return: if the player is winning
    """
    if player == 1:
        for i in range(3, 6):
            for j in range(i - 3, 3):
                if grid[i][j] == 0:
                    return False
    elif player == 2:
        for i in range(10, 13):
            for j in range(i - 3, 10):
                if grid[i][j] == 0:
                    return False
    elif player == 3:
        for i in range(3, 6):
            for j in range(i + 4, 10):
                if grid[i][j] == 0:
                    return False
    return True


def whos_winning(grid):
    """
    Given a board state (oriented normally), heuristically determine
    which player is doing the best. This method returns the player whose pieces
    have the minimum average manhattan distance to their goal
    :param grid: Current board state
    :return: number of player who is faring the best according to the heuristic
    """
    scores = [0, 0, 0]
    for player in [2, 0, 1]:
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == player % 3 + 1:
                    scores[player] += j - i + 5
        grids.rotate_right(grid)
    min_score = np.min(scores)
    scores = (scores - min_score + 1)
    return scores / np.sum(scores)


def search_for_best(grid, piece_locs, k):
    """
    Given a distribution over a grid, search for the sequence of k
    moves that will improve player position the most

    :param grid: The 7x7 game grid
    :param piece_locs: Correspond to the piece to move for each level of the distribution
    :param k: the number of moves deep to search
    :return: the next best move, and how much distance will be gained
    total over the next k moves, as a quadruple (index of piece to move,
    x of destination, y of destination, distance to gain)
    """
    ans = None
    best_gain = -1
    for i, (st_x, st_y) in enumerate(piece_locs):
        valid_spots = all_valid(grid, st_x, st_y)
        for end_x in range(len(grid)):
            for end_y in range(len(grid[0])):
                if valid_spots[end_x][end_y]:
                    if k == 1:
                        gain = score_improvement(st_x, st_y, end_x, end_y)
                    else:
                        piece_locs[i] = (end_x, end_y)
                        move(grid, st_x, st_y, end_x, end_y)
                        _, _, _, gain = search_for_best(grid, piece_locs,
                                                        k - 1)
                        move(grid, end_x, end_y, st_x, st_y)
                        piece_locs[i] = (st_x, st_y)
                    if gain > best_gain:
                        best_gain = gain
                        ans = (i, end_x, end_y, gain)
    assert ans is not None
    return ans


def score_improvement(st_x, st_y, end_x, end_y):
    """
    :return: The distance you will gain to the exit by making the given move
    """
    return (end_y - st_y) - (st_x - end_x)
