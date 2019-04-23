from Data import grids
from Game import logic
from NeuralNetwork import network_utilities
import torch
import numpy as np
from Data import settings

transition_wins = {}

state_seen = set()
transition_seen_count = {}

possible_transitions = {}
prior_probabilities = {}

explore_exploit_constant = .4
temperature = .4

cur_position_hash = 0


def get_train_data(neural_net, to_reset):
    """
    Play some games with MCTS
    Then add the data to help train the network
    :param neural_net: The neural network to help guide the simulation
    :param to_reset: whether to reset the stored weights
    :return: a map from grid state to policy for a few moves
    """
    if to_reset:
        reset()
    updated_policy_keys = []
    updated_policy_values = []
    for game_num in range(1, settings.GAMES_PER_TRAINING + 1):
        reset()
        print("Game " + str(game_num))
        print("Move ", end='')
        grid = grids.get_start_grid()
        cur_player = 3
        hash_grid(grid)
        for move_num in range(1, settings.MAX_MOVES + 1):
            cur_position = cur_position_hash + cur_player
            if move_num % 20 == 1:
                print()
            print(str(move_num) + " ", end='', flush=True)

            set_prior_probabilities(grid, cur_player, neural_net)
            times_to_simulate = len(
                possible_transitions[cur_position]) * settings.MEAN_VISIT_COUNT

            # simulate using MTCS
            for j in range(times_to_simulate, 0, -1):
                rollout(grid, cur_player, neural_net, 0, j)

            # Check probabilities
            move_seen_times = transition_seen_count[cur_position]
            print(sorted(move_seen_times.values()))
            fitnesses = np.zeros(len(move_seen_times))
            for i, move in enumerate(move_seen_times):
                fitnesses[i] = np.power(move_seen_times[move], 1 / temperature)
            check_hash(grid)
            assert 0 not in move_seen_times.values()

            fitnesses /= np.sum(fitnesses)

            # Store this grid of training data
            grids.rotate_to_player(grid, 3, cur_player)
            features, locations = network_utilities.parse_grid(grid,
                                                               cur_player)
            updated_policy_keys.append(features)
            updated_policy_values.append(
                network_utilities.parse_policy(
                    grid, cur_player, locations,
                    possible_transitions[cur_position_hash + cur_player],
                    fitnesses))
            grids.rotate_to_player(grid, cur_player, 3)
            # Go to next move
            old_x, old_y, new_x, new_y = get_move(cur_player, 1)
            # print(str(old_x) + " " + str(old_y) +
            # " " + str(new_x) + " " + str(new_y))
            update_hash(cur_player, old_x, old_y, new_x, new_y)
            logic.move(grid, old_x, old_y, new_x, new_y)

            if logic.is_winning(grid, cur_player):
                print("GAME WON BY " + str(cur_player))
                break
            cur_player = cur_player % 3 + 1
        print()
    return updated_policy_keys, updated_policy_values


def find_good_move(grid, cur_player, neural_net):
    # reset()
    hash_grid(grid)
    # Simulate a bunch
    set_prior_probabilities(grid, cur_player, neural_net)
    times_to_simulate = len(
        possible_transitions[
            cur_position_hash + cur_player]) * settings.THINK_TIME
    for j in range(times_to_simulate, 0, -1):
        rollout(grid, cur_player, neural_net, 0, j)
    return get_move(cur_player, 1)


def rollout(grid, cur_player, neural_net, depth, times_to_simulate):
    """
    Search through the tree of seen possibilities until you get to a leaf,
    then do a rollout to a depth of d and update the
    monte carlo tree probabilities
    :param grid: DA GAME GRID (big, oriented normally)
    :param cur_player: Player to move
    :param neural_net: the network to get prior probabilities with
    :param depth: the depth the rollout searches for a win
    :param times_to_simulate: number of simulations left
    :return:  the result of the simulation. If a player wins, they get 1 point
    and the rest get 0. Shape is (p1points, p2points, p3points) who won in this simulation
    """
    if depth == settings.MAX_DEPTH:
        # estimate who is best and update state win/lose
        return logic.whos_winning(grid)
    return_value = np.zeros(3)
    if logic.is_winning(grid, cur_player):
        return_value[cur_player - 1] = 1
        return return_value
    if (cur_position_hash + cur_player) not in state_seen:
        set_prior_probabilities(grid, cur_player, neural_net)
        # make whatever move the network says to
        move = get_move_from_nn_dist(cur_player)
    else:
        move = get_move(cur_player, times_to_simulate)
    old_x, old_y, new_x, new_y = move
    # see who wins when choosing this move
    cur_hash = cur_position_hash + cur_player
    update_hash(cur_player, old_x, old_y, new_x, new_y)
    logic.move(grid, old_x, old_y, new_x, new_y)
    # print(str(old_x) + " " + str(old_y)
    #      + " " + str(new_x) + " " + str(new_y))
    if transition_seen_count[cur_hash][move] != 0 and depth > 0:
        # went to same position twice - nobody wins
        sim_result = np.array([0, 0, 0])
        transition_seen_count[cur_hash][move] += 1
    else:
        if transition_seen_count[cur_hash][move] == 0:
            depth += 1
            # Backpropogate results
        transition_seen_count[cur_hash][move] += 1
        sim_result = rollout(grid, cur_player % 3 + 1, neural_net,
                             depth, times_to_simulate)
    assert not (sim_result is None)
    update_hash(cur_player, new_x, new_y, old_x, old_y)
    logic.move(grid, new_x, new_y, old_x, old_y)
    transition_wins[cur_hash][move] += sim_result[
        cur_player - 1]
    return sim_result


def set_prior_probabilities(grid, cur_player, neural_net):
    """
    Set the probabilities the network wants to use
    :param grid: The 13 by 13 grid
    :param cur_player: The player to move
    :param neural_net: The network for estimation`
    """
    transition_seen_count[cur_position_hash + cur_player] = {}
    transition_wins[cur_position_hash + cur_player] = {}

    state_seen.add(cur_position_hash + cur_player)
    grids.rotate_to_player(grid, 3, cur_player)
    parsed_grid, locations = network_utilities.parse_grid(grid, cur_player)
    data = torch.from_numpy(parsed_grid).view(-1, 9, 7, 7).float()

    distribution = neural_net.forward(data)[0].detach().numpy()
    distribution = network_utilities.ignore_impossible_moves(grid,
                                                             locations,
                                                             distribution)
    grids.rotate_to_player(grid, cur_player, 3)
    move_list = []
    probability_list = []
    for i in range(len(distribution)):
        st_i, st_j = grids.rotate_move_to_player(
            cur_player, 3, locations[i][0] + 3, locations[i][1] + 3)
        for end_i in range(len(distribution[0])):
            for end_j in range(len(distribution[0][0])):
                if distribution[i][end_i][end_j] > 0:
                    new_end_i, new_end_j = grids.rotate_move_to_player(
                        cur_player, 3, end_i + 3, end_j + 3)
                    move_list.append((st_i, st_j,
                                      new_end_i, new_end_j))
                    probability_list.append(distribution[i][end_i][end_j])

    possible_transitions[cur_position_hash + cur_player] = move_list
    probability_list = np.array(probability_list)
    prior_probabilities[cur_position_hash + cur_player] = probability_list
    for move in possible_transitions[cur_position_hash + cur_player]:
        transition_seen_count[cur_position_hash + cur_player][move] = 0
        transition_wins[cur_position_hash + cur_player][move] = 0


def get_move(cur_player, times_to_simulate):
    """
    Get the best next move based on the prior probability P, number of times
    the position has been seen S, number of wins seen W, and times left
    to simulate.
    the 'best' move maximizes 1/(P+S) + W/S + c*sqrt(log(T)/S)
    :param cur_player: The player currently moving
    :param times_to_simulate: The number of simulations left
    :return: A move (stx, sty, endx, endy)
    """
    move_list = possible_transitions[cur_position_hash + cur_player]
    probability_list = prior_probabilities[
        cur_position_hash + cur_player]
    # probability_list += np.random.dirichlet(
    #    len(probability_list) * [.5]) * .25
    fitnesses = []
    max_fitness = 0
    argmax_fitness = None
    for i in range(len(probability_list)):
        win_count = transition_wins[cur_position_hash + cur_player][
            move_list[i]]
        seen_count = transition_seen_count[cur_position_hash + cur_player][
            move_list[i]]
        fitness_i = fitness(probability_list[i], seen_count, win_count,
                            times_to_simulate, len(probability_list))
        if fitness_i > max_fitness:
            max_fitness = fitness_i
            argmax_fitness = i
        fitnesses.append(fitness_i)
    return move_list[argmax_fitness]


def fitness(prior_probability, seen_count, win_count, times_to_simulate,
            total_moves):
    """
    Given an state, and prior probability of moving to that state,
    calculates the 'fitness' of the state.
    :param prior_probability: Chance of moving to this position according
    to the neural network
    :param seen_count: number of times this move has been seen
    :param win_count: number of times won from this position
    :param times_to_simulate: times left to simulate
    :return: Fitness of the given move
    """
    # according to Mastering the Game of Go without Human Knowledge
    move_fitness = 2 * total_moves * prior_probability / (1 + seen_count)
    if seen_count != 0:
        # according to
        # https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
        move_fitness += (win_count / seen_count) + \
                        np.multiply(explore_exploit_constant,
                                    np.sqrt(np.log(
                                        times_to_simulate) / seen_count))
    else:
        move_fitness += 10
    return move_fitness


def get_move_from_nn_dist(cur_player):
    """
    Returns a move based on the distribution stored in the NN
    :param position: (hash(grid), cur_player) tuple)
    :return: A move (stx, sty, endx, endy)
    """
    move_list = possible_transitions[cur_position_hash + cur_player]
    probability_list = prior_probabilities[cur_position_hash + cur_player]
    assert .99999 <= sum(probability_list) <= 1.00001
    rand = np.random.random()
    for i, p in enumerate(probability_list):
        rand -= p
        if rand <= 0:
            return move_list[i]
    return get_move_from_nn_dist(cur_player)


def hash_grid(grid):
    """
    Stores integers representing the game state in cur_position_hash
    :param grid: the 13 by 13 current game state
    """
    grid_size = len(grid)
    assert grid_size == 13
    hashVal = 0
    for i in range(0, grid_size):
        for j in range(0, grid_size):
            if grid[i][j] != 9:
                hashVal += int(grid[i][j]) * (4 ** (1 + i + grid_size * j))
    global cur_position_hash
    cur_position_hash = hashVal


def update_hash(player, old_x, old_y, new_x, new_y):
    """
    Update all the hashes when a move is made
    """
    global cur_position_hash
    cur_position_hash -= player * (4 ** (1 + old_x + 13 * old_y))
    cur_position_hash += player * (4 ** (1 + new_x + 13 * new_y))


def check_hash(grid):
    grid_size = 13
    hash_val = 0
    for i in range(0, grid_size):
        for j in range(0, grid_size):
            if grid[i][j] != 9:
                hash_val += int(grid[i][j]) * (4 ** (1 + i + grid_size * j))
    assert hash_val == cur_position_hash


def reset():
    """
    Reset all of the global variables for this function
    """
    state_seen.clear()
    transition_wins.clear()
    transition_seen_count.clear()
    possible_transitions.clear()
    prior_probabilities.clear()
