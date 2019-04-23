import torch
import torch.nn as nn

in_channels = 3
out_channels = 10
grid_height = 7
grid_width = 7
n_pieces = 6


class CheckersNetworkSmall(nn.Module):
    """
    A class intended to train a neural network to play
    6-piece chinese checkers according to the 8x8 square
    in the center of the network
    """

    def __init__(self):
        """
        Initialize the network, creating layers and stuff
        """
        super(CheckersNetworkSmall, self).__init__()
        self.sequential_model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding=1,
                      kernel_size=(3, 3)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels,
                      padding=1, kernel_size=(3, 3)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels,
                      padding=1, kernel_size=(3, 3)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels,
                      padding=1, kernel_size=(3, 3)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels,
                      padding=1, kernel_size=(3, 3)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.layer_convolution = nn.Sequential(
            nn.Conv2d(out_channels + 1, out_channels, padding=1,
                      kernel_size=(3, 3)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.LinearFinisher = nn.Linear(
            out_channels * grid_height * grid_width,
            grid_height * grid_width)

    def forward(self, input_tensor):
        """
        :param: input_tensor: a (Batch Size) by 9 by 7 by 7 vector representing
        piece locations
        :return: a (Batch Size) by 6 by 7 by 7 vector representing move
        probability distributions
        """

        # First, convolute the layers a bunch
        temp, p1, p2, p3, p4, p5, p6 = torch.split(input_tensor,
                                                   [3, 1, 1, 1, 1, 1, 1],
                                                   1)
        # out = out.view(-1,
        #               out_channels * grid_height * grid_width)
        temp = self.sequential_model(temp)
        p1 = self.layer_convolution(torch.cat((temp, p1), 1))
        p2 = self.layer_convolution(torch.cat((temp, p2), 1))
        p3 = self.layer_convolution(torch.cat((temp, p3), 1))
        p4 = self.layer_convolution(torch.cat((temp, p4), 1))
        p5 = self.layer_convolution(torch.cat((temp, p5), 1))
        p6 = self.layer_convolution(torch.cat((temp, p6), 1))
        size = out_channels * grid_height * grid_width
        p1 = self.LinearFinisher(p1.view(-1, size))
        p2 = self.LinearFinisher(p2.view(-1, size))
        p3 = self.LinearFinisher(p3.view(-1, size))
        p4 = self.LinearFinisher(p4.view(-1, size))
        p5 = self.LinearFinisher(p5.view(-1, size))
        p6 = self.LinearFinisher(p6.view(-1, size))
        temp = torch.cat((p1, p2, p3, p4, p5, p6), 1)
        out = nn.Softmax(-1)(temp)
        out = out.view(-1, n_pieces, grid_height, grid_width)
        return out
