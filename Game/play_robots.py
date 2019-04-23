import numpy as np
import pygame
import torch

from Data import settings, grids
from Game import logic, check_valid_moves
from Graphics import graphics
from NeuralNetwork import mcts
from NeuralNetwork.checkers_network import CheckersNetwork
from NeuralNetwork.checkers_network_small import CheckersNetworkSmall
from NeuralNetwork.network_utilities import get_max_move, get_move
from time import sleep

PATH0 = ""  # Red
PATH1 = "../Data/small_net.pkl"  # Blue
PATH2 = "../Data/overnight_net.pkl"  # Yellow

if __name__ == '__main__':

    cur_player = 1
    model = [CheckersNetworkSmall(), CheckersNetworkSmall(),
             CheckersNetworkSmall()]

    # model[0].load_state_dict(torch.load(PATH0))
    # model[0].eval()

    model[1].load_state_dict(torch.load(PATH1))
    model[1].eval()

    model[2].load_state_dict(torch.load(PATH2))
    model[2].eval()

    moves = 0
    winners = [0, 0, 0]
    show_display = False
    for i in range(90):
        grid = np.array(graphics.START_GRID)
        if show_display:
            screen = graphics.show_grid(grid)
        lastPos = []
        print(i, end=' ', flush=True)
        print(moves)
        moves = 0
        while True:
            if show_display:
                for event in pygame.event.get():  # User did something
                    if event.type == pygame.QUIT:  # If user clicked close
                        exit(0)
                graphics.display_cur_move(screen, cur_player)
            if settings.THINK_TIME == 0:
                grids.rotate_to_player(grid, 3, cur_player)
                if moves > 30:
                    sti, stj, endi, endj = get_max_move(grid,
                                                        model[cur_player - 1],
                                                        cur_player)
                else:
                    sti, stj, endi, endj = get_move(grid,
                                                    model[0],
                                                    cur_player)
                sti, stj = grids.rotate_move_to_player(cur_player, 3, sti,
                                                       stj)
                endi, endj = grids.rotate_move_to_player(cur_player, 3,
                                                         endi,
                                                         endj)
                grids.rotate_to_player(grid, cur_player, 3)
            else:
                sti, stj, endi, endj = mcts.find_good_move(grid,
                                                           cur_player,
                                                           model[
                                                               cur_player - 1])
            assert check_valid_moves.is_valid(grid, sti, stj, endi, endj)

            if show_display:
                graphics.move(screen, grid, sti, stj, endi, endj)
            logic.move(grid, sti, stj, endi, endj)
            if show_display:
                if logic.update_gui_if_player_winning(screen, grid,
                                                      cur_player):
                    winners[cur_player - 1] += 1
                    break
            elif logic.is_winning(grid, cur_player):
                winners[cur_player - 1] += 1
                break
            cur_player = cur_player % 3 + 1
            moves += 1
            # sleep(.1)
    print(winners)
