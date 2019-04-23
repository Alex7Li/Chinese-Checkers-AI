import pygame
import math
from Data import grids
from Data import settings
from Data.settings import SQ_SIZE
from time import sleep

# Rows of pieces that each player has
ROWS = 3
START_GRID = grids.get_start_grid(ROWS)

RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
LIGHTRED = (155, 0, 0)
LIGHTBLUE = (0, 0, 155)
LIGHTYELLOW = (155, 155, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
color_dict = {0: WHITE,
              1: RED,
              2: BLUE,
              3: YELLOW,
              4: LIGHTRED,
              5: LIGHTBLUE,
              6: LIGHTYELLOW,
              9: BLACK}
player_map = {
    1: 'red',
    2: 'blue',
    3: 'yellow'
}
# Number of squares on a side of the grid
GRID_SIZE = len(START_GRID)
MESSAGE_BOARD_SIZE = 30


def move(screen, grid, old_i, old_j, new_i, new_j):
    """
    Updates the screen, moving a rectangle from
    (oldI, oldJ) to  (newI, newJ)
    :param screen: Screen viewer looks at
    :param grid: Full board state
    :param old_i: i-index of old point
    :param old_j: j-index of old point
    :param new_i: i-index of new point
    :param new_j: j-index of new point
    """
    if settings.IS_RECT:
        rects = [draw_rect(screen, old_i, old_j, 0),
                 draw_rect(screen, new_i, new_j, grid[old_i][old_j])]
    else:
        rects = [draw_circ(screen, old_i, old_j, 0),
                 draw_circ(screen, new_i, new_j, grid[old_i][old_j])]
    pygame.display.update(rects)


def draw_rect(screen, i, j, color):
    """
    Draws the appropriate rectangle to the screen
    Returns a rectangle containing the drawn rectangle
    :param screen: Screen viewer looks at
    :param i: i-index of rectangle
    :param j: j-index of rectangle
    :param color: color to make the rectange
    """
    rect = (i * SQ_SIZE, j * SQ_SIZE, SQ_SIZE, SQ_SIZE)
    pygame.draw.rect(screen, color_dict[color], rect)
    if color_dict[color] == WHITE:
        pygame.draw.rect(screen, BLACK, rect, 1)
    return pygame.Rect(rect)


def draw_circ(screen, i, j, color):
    """
    Draws the appropriate circle to the screeen
    :param screen: Screen viewer looks at
    :param i: i-index of circle
    :param j: j-index of circle
    :param color: color to make the circle
    """
    # Formulas which ensure that the coordinates fill the screen correctly
    x_offset = int((i - j + GRID_SIZE / 2) * SQ_SIZE)
    y_offset = int((i + j - ROWS + 1) * SQ_SIZE / math.sqrt(3))
    radius = SQ_SIZE // 3
    center = [x_offset, y_offset]
    pygame.draw.circle(screen, color_dict[color], center, radius)
    return pygame.Rect((x_offset - radius), (y_offset - radius), 2 * radius,
                       2 * radius)


def get_square_at(x, y):
    """
    Get the index of the square at a given location on the screen
    :param x: pixels from right
    :param y: pixels from top
    :return: (i index, j index)
    """
    if settings.IS_RECT:
        i = x // SQ_SIZE
        j = y // SQ_SIZE
    else:
        i = (y * math.sqrt(3) /
             SQ_SIZE + x / SQ_SIZE - GRID_SIZE / 2 + ROWS) / 2
        j = i - x / SQ_SIZE + GRID_SIZE / 2
    return int(i), int(j)


def highlight(screen, grid, i, j):
    """
    Highlight a piece on the screen
    :param screen: screen to highlight on
    :param grid: current game state
    :param i: i-index of piece
    :param j: j-index of piece
    """
    assert not (grid[i][j] == 9 or grid[i][j] == 0)
    grid[i][j] += 3
    if settings.IS_RECT:
        update_rect = draw_rect(screen, i, j, grid[i][j])
    else:
        update_rect = draw_circ(screen, i, j, grid[i][j])
    pygame.display.update(update_rect)


def unhighlight(screen, grid, i, j):
    """
    Unhighlight a highlighted piece on the screen
    :param screen: screen to highlight on
    :param grid: current game state
    :param i: i-index of piece
    :param j: j-index of piece
    """
    assert 4 <= grid[i][j] <= 6
    grid[i][j] -= 3
    if settings.IS_RECT:
        update_rect = draw_rect(screen, i, j, grid[i][j])
    else:
        update_rect = draw_circ(screen, i, j, grid[i][j])
    pygame.display.update(update_rect)


def show_grid(grid):
    """
    Create a screen that displays the given grid,
    and display it to the used
    :param grid: The current game state
    :return: A screen for the user to see
    """
    pygame.init()
    if settings.IS_RECT:
        screen_size = [SQ_SIZE * GRID_SIZE,
                       SQ_SIZE * GRID_SIZE + MESSAGE_BOARD_SIZE]
    else:
        screen_size = [SQ_SIZE * GRID_SIZE,
                       SQ_SIZE * (GRID_SIZE - int(
                           ROWS / math.sqrt(3))) + MESSAGE_BOARD_SIZE]
    screen = pygame.display.set_mode(screen_size)
    pygame.display.set_caption("Chinese Checkers")
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if settings.IS_RECT:
                draw_rect(screen, i, j, grid[i][j])
            else:
                draw_circ(screen, i, j, grid[i][j])
    pygame.display.update()
    return screen


def write_message(screen, message):
    """
    Writes a message in the message board of the screen
    :param screen: The screen to write to
    :param message: The text the message should contain
    """
    myfont = pygame.font.SysFont('Arial', 30)
    textsurface = myfont.render(message, False, WHITE)
    rect = (10, get_height() - 10, SQ_SIZE * GRID_SIZE, MESSAGE_BOARD_SIZE)
    pygame.draw.rect(screen, BLACK, rect)
    screen.blit(textsurface, (10, get_height() - 10))
    pygame.display.update(rect)


def get_height():
    """
    :return: The height of the game screen, ignoring the message board
    """
    if settings.IS_RECT:
        screen_height = SQ_SIZE * GRID_SIZE
    else:
        screen_height = SQ_SIZE * (GRID_SIZE - int(ROWS / math.sqrt(3)))
    return screen_height


def display_victory(screen, winning_player):
    """
    Write who wins to the screen (:
    :param screen: The screen to write to
    :param winning_player: The number of the player that won
    """
    winner_name = player_map[winning_player]
    write_message(screen, "The winner is " + str(winner_name))


def display_cur_move(screen, cur_player):
    """
    Write whose turn it currently is to the screen
    :param screen: The screen to write to
    :param cur_player: The player whose turn it is
    """
    player_name = player_map[cur_player]
    write_message(screen, str(player_name) + " to move")


if __name__ == '__main__':
    show_grid(START_GRID)
    done = False
    while not done:
        for event in pygame.event.get():  # User did something
            if event.type == pygame.QUIT:  # If user clicked close
                done = True  # Flag that we are done so we exit this loop
