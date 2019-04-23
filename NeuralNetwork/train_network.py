import os

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

from Data import grids
from Game import logic, check_valid_moves
from NeuralNetwork.checkers_network_small import CheckersNetworkSmall
from NeuralNetwork.mcts import get_train_data
from NeuralNetwork.network_utilities import get_move

LOAD_PATH = "../Data/overnight_net_2.pkl"
SAVE_PATH = "../Data/overnight_net_2.pkl"

num_epochs = 500
batch_size = 50


def train(cur_model, iterations):
    """
    Train the neural network and save it
    :param cur_model: The current NN model
    :param iterations: The number of times to retrain the NN
    """
    reset = False
    for iteration in range(1, 1 + iterations):
        learning_rate = .1
        print("iteration " + str(iteration))
        train_keys, train_values = get_train_data(cur_model, reset)
        train_features = torch.DoubleTensor(train_keys)
        train_labels = torch.FloatTensor(train_values)
        print("Train size: " + str(len(train_features)))
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(cur_model.parameters(), lr=learning_rate)
        last_epochs_loss = 100
        cur_epochs_loss = 0
        for epoch in range(1, num_epochs + 1):
            for i in range(0, len(train_labels), batch_size):
                features = Variable(train_features[i:i + batch_size]).float()
                labels = Variable(train_labels[i:i + batch_size]).float()
                # Initialize the hidden weight to all zeros
                optimizer.zero_grad()
                outputs = cur_model(features)
                # Compute the loss: difference
                # between the output class and the pre-given label
                loss = criterion(outputs, labels)
                loss.backward()  # Backward pass: compute the weight
                optimizer.step()  # Update the weights of hidden nodes
                cur_epochs_loss += loss.data.item()
            if epoch % 10 == 0:
                print("%.7f" % last_epochs_loss, end=' ')
                if cur_epochs_loss > last_epochs_loss:
                    learning_rate /= 2
                    if learning_rate < .005:
                        break
                last_epochs_loss = cur_epochs_loss
                cur_epochs_loss = 0
                if epoch % 7 == 0:
                    print()
        print("Done training. Testing vs old model")
        if os.path.isfile(SAVE_PATH):
            old_model = CheckersNetworkSmall()
            old_model.load_state_dict(torch.load(SAVE_PATH))
            old_model.eval()
            cur_model = compare_model(old_model, cur_model)
        torch.save(cur_model.state_dict(), SAVE_PATH)
        print("Training Iteration over")


def compare_model(old_model, cur_model):
    wins = 0
    cur_player = 2
    games = 81
    for game in range(1, games + 1):
        grid = np.array(grids.get_start_grid())
        for moves in range(300):
            grids.rotate_to_player(grid, 3, cur_player)
            if (cur_player + game) % 2 == 0:
                sti, stj, endi, endj = get_move(grid, cur_model, cur_player)
            else:
                sti, stj, endi, endj = get_move(grid, old_model, cur_player)
            sti, stj = grids.rotate_move_to_player(cur_player, 3, sti, stj)
            endi, endj = grids.rotate_move_to_player(cur_player, 3, endi,
                                                     endj)
            grids.rotate_to_player(grid, cur_player, 3)
            assert check_valid_moves.is_valid(grid, sti, stj, endi, endj) == 1

            logic.move(grid, sti, stj, endi, endj)
            if logic.is_winning(grid, cur_player):
                if (cur_player + game) % 2 == 0:
                    wins += 1
                wins -= .5
                break
            cur_player = cur_player % 3 + 1
        wins += .5
    print(f"Won {wins} matches of {games} in the tournament")
    if wins >= games // 2:
        print("Using new model")
        return cur_model
    else:
        print("Using old model")
        return old_model


def test_shapes(nn_model):
    """
    Perform a test on the neural network, ensuring
    that the outputs make sense
    """
    st_grid = grids.get_start_grid(4)
    get_train_data(nn_model, False)
    data = torch.from_numpy(st_grid).view(-1, 9, 7, 7).float()
    res = nn_model.forward(data)

    print(res.shape)
    print(np.count_nonzero(res > .00001))
    print(np.sum(res.detach().numpy()))
    print(res)


if __name__ == '__main__':
    model = CheckersNetworkSmall()
    #assert 1 == 2
    if os.path.isfile(LOAD_PATH):
        model.load_state_dict(torch.load(LOAD_PATH))
        model.eval()
    train(model, 200)
