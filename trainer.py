import random
import socket
import torch
import json

from torch import nn

assert torch.cuda.is_available()

EPISODE_COUNT = 500
BOARD_WIDTH = None
BOARD_HEIGHT = None
ACTION_COUNT = 6


# Game State class
class State:
    def __init__(self, json_string):
        self.data = json.loads(json_string)

        global BOARD_WIDTH
        if BOARD_WIDTH == None:
            BOARD_WIDTH = len(self.data['players'][0]['tiles']['board'])

        global BOARD_HEIGHT
        if BOARD_HEIGHT is None:
            BOARD_HEIGHT = len(self.data['players'][0]['tiles']['board'][0])


    def GameOver(self):
        return self.data['gameOver']


    def GetPerception(self, playerIndex):
        perception = torch.zeros([BOARD_WIDTH, BOARD_HEIGHT, 4], dtype=torch.float)
        for x in range(BOARD_WIDTH):
            for y in range(BOARD_HEIGHT):
                tile = self.data['players'][playerIndex]['tiles']['board'][x][y]

                if tile != 0:
                    perception[x][y][tile - 1] = 1

        return perception



# Model class
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.own_tiles_convolution = nn.Sequential(
            nn.Conv2d(4, 8, 5, padding='same'),
            nn.ReLU(),
            nn.Conv2d(8, 8, 5, padding='same'),
            nn.ReLU(),
            nn.Conv2d(8, 8, 5, padding='same'),
            nn.ReLU(),
        )

        self.enemy_tiles_convolution = nn.Sequential(
            nn.Conv2d(1, 8, 5),
            nn.ReLU(),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(2 * 8 * BOARD_WIDTH * BOARD_HEIGHT, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 6),
            nn.ReLU(),
            nn.Softmax(),
        )

    def forward(self, x):
        # Split input into our board and the enemy board
        our_board, their_board = x.split([4 * BOARD_WIDTH * BOARD_HEIGHT, 4 * BOARD_WIDTH * BOARD_HEIGHT])

        our_board = our_board.reshape((BOARD_WIDTH, BOARD_HEIGHT, 4))
        our_board = our_board.permute((2, 0, 1))
        our_board = self.own_tiles_convolution(our_board.cuda())

        their_board = their_board.reshape((BOARD_WIDTH, BOARD_HEIGHT, 4))
        their_board = their_board.permute((2, 0, 1))
        their_board = self.own_tiles_convolution(their_board.cuda())

        combined = torch.cat([our_board.flatten(), their_board.flatten()])

        actions = self.linear_layers(combined)

        return actions


# Initialise model
MODEL = None

# Connect to Puyo-Puyo server
SERVER = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
SERVER.connect(("127.0.0.1", 25500))


def RandomActions():
    return [random.randint(0, ACTION_COUNT - 1), random.randint(0, ACTION_COUNT - 1)]


def SelectActions(state):
    perception = [
        torch.cat([torch.flatten(state.GetPerception(0)), torch.flatten(state.GetPerception(1))]),
        torch.cat([torch.flatten(state.GetPerception(1)), torch.flatten(state.GetPerception(0))])
    ]

    outputs = [MODEL(p) for p in perception]

    choices = [int(o.max(0).indices.view(1, 1)) for o in outputs]

    return choices


for i in range(EPISODE_COUNT):
    while True:
        state = State(SERVER.recv(4096))

        if MODEL is None:
            MODEL = Network().to("cuda")

        if state.GameOver():
            continue

        selected_actions = SelectActions(state)
        SERVER.send(bytearray(selected_actions))
