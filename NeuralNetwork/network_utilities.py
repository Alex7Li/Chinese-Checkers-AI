import numpy as np
import torch
from Data import grids, settings
from Game.logic import search_for_best
from Game import check_valid_moves


def parse_grid(grid, cur_player):
    """
    Compress the grid and make it suitable for the neural network
    :param grid: The current 13x13 game state, rotated to
    show cur_player's position
    :param cur_player: The player whose move it is
    :return: The features map to be used for the neural net, the
    locations of the pieces corresponding to the output layers
    """

    # Remove less useful features to shrink size (169 cells->49 cells)
    input = grid[3:10, 3:10]
    # Feature map is 3+6 levels. The first 3 levels are the positions
    # of pieces. The next 6 levels are the locations of each
    # friendly piece.
    feature_map = np.zeros((9, 7, 7))
    piece_num = 0
    piece_locs = []
    for i in range(0, 7):
        for j in range(0, 7):
            if input[i][j] != 0:
                player = (input[i][j] - cur_player) % 3
                if player == 0:
                    feature_map[piece_num + 3][i][j] = 1
                    piece_num += 1
                    piece_locs.append((i, j))
                feature_map[player][i][j] = 1
    assert piece_num == 6
    return feature_map, piece_locs


def parse_policy(grid, cur_player, locations, moves, move_probabilities):
    """
    Transform a policy from mcts into the shape expected from the neural
    network
    :param grid: The current 13x13 grid state (oriented normally)
    :param cur_player: The current player
    :param locations: The ordered locations from the NN (oriented normally)
    :param moves: The moves to use (oriented to player)
    :param move_probabilities: chance taking each move (policy from mcts)
    :return: A NN-compatible output representing the policy
    """
    expected_output = np.zeros([6, 7, 7])
    assert len(moves) == len(move_probabilities)
    for ind, move in enumerate(moves):
        oldi, oldj, newi, newj = move
        oldi, oldj = grids.rotate_move_to_player(3, cur_player, oldi, oldj)
        newi, newj = grids.rotate_move_to_player(3, cur_player, newi, newj)
        assert cur_player == grid[oldi][oldj]
        from_ind = locations.index((oldi - 3, oldj - 3))
        expected_output[from_ind][newi - 3][newj - 3] = move_probabilities[ind]
    return expected_output


def ignore_impossible_moves(grid, piece_locs, distribution):
    """
    Given a distribution over a grid, remove all locations you can't get to
    and normalize to 1. This method removes locations based on the point of
    view of the neural network, so it may consider extra spots to be
    impossible. Additionally, if NO_BACKSIES is set to true, it will consider
    spaces where the AI goes backwards to be imposisble

    :param grid: The whole 13x13 grid
    :param piece_locs: Correspond to the piece to move for each level of the distribution
    :param distribution: Network output
    :return: distribution without impossibilities, normalized to sum to 1
    """
    grid = grid[3:10, 3:10]
    # print(distribution)
    assert len(distribution) == 6
    for i in range(len(distribution)):
        valid_spots = check_valid_moves.all_valid(grid, piece_locs[i][0],
                                                  piece_locs[i][1])
        distribution[i] = np.where(valid_spots, distribution[i], 0)
        if settings.NO_BACKSIES:
            st_score = piece_locs[i][1] - piece_locs[i][0]
            for j in range(len(distribution[0])):
                for k in range(len(distribution[0])):
                    if k - j <= st_score:
                        distribution[i][j][k] = 0
                    elif valid_spots[j][k] and distribution[i][j][k] == 0:
                        print("We have ignored a valid position!")
                        print(grid)
                        print(piece_locs[i])
                        print(str(j) + " " + str(k))
                        print(distribution[i])
    if np.sum(distribution) == 0:
        # going forward blindly doesn't work ):
        piece, x, y, _ = search_for_best(grid, piece_locs, 2)
        distribution[piece][x][y] = 1
    return np.divide(distribution, np.sum(distribution))


def get_move(grid, model, cur_player):
    """
    :param grid: The current 13 by 13 game state
    :param model: The neural network
    :param cur_player: The player currently moving
    :return: A random move according to the model's probability
    distribution
    """
    parsed_grid, locations = parse_grid(grid, cur_player)
    data = torch.from_numpy(parsed_grid).view(-1, 9, 7, 7).float()
    distribution = model.forward(data)[0].detach().numpy()
    distribution = ignore_impossible_moves(grid, locations,
                                           distribution)
    assert np.sum(0 > distribution) == 0
    assert .999 <= np.sum(distribution) <= 1.001

    rand = np.random.random()
    for i in range(len(distribution)):
        for j in range(len(distribution[i])):
            for k in range(len(distribution[i][j])):
                rand -= max(0, distribution[i][j][k])
                if rand <= 0:
                    return locations[i][0] + 3, locations[i][
                        1] + 3, j + 3, k + 3
    return get_move(grid, model, cur_player)


def get_max_move(grid, model, cur_player):
    """
    :param grid: The current 13 by 13 game state
    :param model: The neural network
    :param cur_player: The player currently moving
    :return: the move with the highest probability of being best
    according to the model
    """
    parsed_grid, locations = parse_grid(grid, cur_player)
    data = torch.from_numpy(parsed_grid).view(-1, 9, 7, 7).float()
    distribution = model.forward(data)[0].detach().numpy()
    distribution = ignore_impossible_moves(grid, locations,
                                           distribution)
    assert np.sum(0 > distribution) == 0

    bestP = 0
    bestMove = (-1, -1, -1, -1)
    for i in range(len(distribution)):
        for j in range(len(distribution[i])):
            for k in range(len(distribution[i][j])):
                if distribution[i][j][k] >= bestP:
                    bestP = distribution[i][j][k]
                    bestMove = (locations[i][0] + 3, locations[i][
                        1] + 3, j + 3, k + 3)
    return bestMove
