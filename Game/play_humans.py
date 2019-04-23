import pygame
from Graphics import graphics
from Game import logic
import numpy as np
from Game.check_valid_moves import is_valid, spot_type, is_in_valid_range
from Data import settings

if __name__ == '__main__':
    grid = np.array(graphics.START_GRID)
    screen = graphics.show_grid(grid)

    cur_player = 1
    # The first player is 3. 3 is in a decent viewing spot (going left to
    # right), and this way the NN and human are both thinking from the
    # same perspective, making the code easier to understand
    done = False
    lastPos = []
    while not done:
        for event in pygame.event.get():  # User did something
            if event.type == pygame.QUIT:  # If user clicked close
                done = True  # Flag that we are done so we exit this loop
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                x, y = graphics.get_square_at(pos[0], pos[1])
                print(str(x) + " " + str(y))
                if spot_type(grid, x, y) == -1:
                    continue
                if not lastPos:
                    if grid[x][y] == cur_player or (
                            settings.CHEATS and 1 <= grid[x][y] <= 3):
                        graphics.highlight(screen, grid, x, y)
                        lastPos = (x, y)
                else:
                    if is_valid(grid, lastPos[0], lastPos[1],
                                x, y) and is_in_valid_range(cur_player, x, y):
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
