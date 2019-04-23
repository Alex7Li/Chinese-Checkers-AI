import numpy as np
import pygame
import torch

from Data import settings, grids
from Game import logic
from Game.check_valid_moves import is_valid, spot_type, is_in_valid_range
from Graphics import graphics
from NeuralNetwork import mcts
from NeuralNetwork.checkers_network import CheckersNetwork
from NeuralNetwork.checkers_network_small import CheckersNetworkSmall
from NeuralNetwork.network_utilities import get_max_move

PATH_1 = "../Data/overnight_net.pkl"  # Blue
PATH_2 = "../Data/overnight_net.pkl"  # Yellow

if __name__ == '__main__':
    grid = np.array(graphics.START_GRID)
    screen = graphics.show_grid(grid)

    cur_player = 3
    human_players = 1
    model1 = CheckersNetworkSmall()
    model1.load_state_dict(torch.load(PATH_1))
    model1.eval()
    model2 = CheckersNetworkSmall()
    # model2.load_state_dict(torch.load(PATH_2))
    # model2.eval()

    done = False
    lastPos = []
    while not done:
        for event in pygame.event.get():  # User did something
            if event.type == pygame.QUIT:  # If user clicked close
                done = True  # Flag that we are done so we exit this loop
            if cur_player <= human_players and \
                    event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                x, y = graphics.get_square_at(pos[0], pos[1])
                if spot_type(grid, x, y) == -1:
                    continue
                if not lastPos:
                    if grid[x][y] == cur_player or (
                            settings.CHEATS and 1 <= grid[x][y] <= 3):
                        graphics.highlight(screen, grid, x, y)
                        lastPos = (x, y)
                else:
                    if is_valid(grid, lastPos[0], lastPos[1], x, y) and \
                            is_in_valid_range(cur_player, x, y):
                        graphics.unhighlight(screen, grid, lastPos[0],
                                             lastPos[1])
                        graphics.move(screen, grid, lastPos[0], lastPos[1], x,
                                      y)
                        logic.move(grid, lastPos[0], lastPos[1], x, y)
                        lastPos = []
                        logic.update_gui_if_player_winning(screen, grid,
                                                           cur_player)
                        cur_player = cur_player % 3 + 1
                    elif grid[x][y] == cur_player or (
                            settings.CHEATS and 1 <= grid[x][y] <= 3):
                        graphics.unhighlight(screen, grid, lastPos[0],
                                             lastPos[1])
                        graphics.highlight(screen, grid, x, y)
                        lastPos = (x, y)
        graphics.display_cur_move(screen, cur_player)
        if cur_player > human_players:
            if cur_player == 3:
                model = model1
            else:
                model = model2
            if settings.THINK_TIME == 0:
                grids.rotate_to_player(grid, 3, cur_player)
                sti, stj, endi, endj = get_max_move(grid, model, cur_player)
                sti, stj = grids.rotate_move_to_player(cur_player, 3, sti, stj)
                endi, endj = grids.rotate_move_to_player(cur_player, 3, endi,
                                                         endj)
            else:
                sti, stj, endi, endj = mcts.find_good_move(grid, cur_player,
                                                           model)
            if settings.THINK_TIME == 0:
                grids.rotate_to_player(grid, cur_player, 3)
            assert is_valid(grid, sti, stj, endi, endj)

            graphics.move(screen, grid, sti, stj, endi, endj)
            logic.move(grid, sti, stj, endi, endj)
            logic.update_gui_if_player_winning(screen, grid, cur_player)
            cur_player = cur_player % 3 + 1
