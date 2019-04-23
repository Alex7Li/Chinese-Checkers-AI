# if the grid is rectangular or normal
IS_RECT = False

# If the grid rotates to the player whose turn it is
# ROTATION = False

# If you are allowed to move any player whenever
CHEATS = False

# Factor multiplying how long the AI thinks (or none to just use NN)
# Basically just MEAN_VISIT_COUNT, but for playing instead of training
THINK_TIME = 4

# Edge length of every square in pixels
SQ_SIZE = 50

# The AI will never go backwards, or even to the side (unless it is
# the only choice)
NO_BACKSIES = True

# The average number of times that the mtcs will try each move
# when determining the probability distribution for a given position
MEAN_VISIT_COUNT = 15

# The maximum depth that the mcts will go after reaching a leaf
# node. Lower is more greedy, higher takes longer
MAX_DEPTH = 4

# The maximum number of moves that will be played in each simulation game
MAX_MOVES = 100

# The number of games that will be simulated per time that the network trains
GAMES_PER_TRAINING = 3
