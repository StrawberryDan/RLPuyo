import socket
import torch
import json

from torch import nn

assert torch.cuda.is_available()

EPISODE_COUNT = 500
BOARD_WIDTH = None
BOARD_HEIGHT = None


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


# Model class
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.own_tiles_covolution = nn.Sequential(
            nn.Conv2d(1, 8, 5),
            nn.ReLU(),
        )

        self.enemy_tiles_colvolution = nn.Sequential(
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
        pass


# Initialise model
MODEL = None

# Connect to Puyo-Puyo server
SERVER = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
SERVER.connect(("127.0.0.1", 25500))


def RandomActions():
    return [random.randint(0, ACTION_COUNT - 1), random.randint(0, ACTION_COUNT - 1)]


def SelectActions():
    return RandomActions()


for i in range(EPISODE_COUNT):
    while True:
        state = State(SERVER.recv(4096))

        if MODEL is None:
            MODEL = Network().to("cuda")

        if state.GameOver():
            continue

        selected_actions = SelectActions()
        SERVER.send(bytearray(selected_actions))
