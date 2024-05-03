import datetime
import math
import random
import socket
from collections import deque
import csv
import torch
import json

from torch import nn

assert torch.cuda.is_available()

EPISODE_COUNT = 1000
BATCH_SIZE = 32
BOARD_WIDTH = None
BOARD_HEIGHT = None
ACTION_COUNT = 6 - 2
EXPERIENCE_BUFFER_SIZE = 5000
DISCOUNT_FACTOR = 0.75
EXPLORATION_PROBABILITY = 0.75
EPSILON = EXPLORATION_PROBABILITY


# Game State class
class State:
    def __init__(self, json_string):
        self.data = json.loads(json_string)

        global BOARD_WIDTH
        if BOARD_WIDTH is None:
            BOARD_WIDTH = len(self.data['players']['tiles']['board'])

        global BOARD_HEIGHT
        if BOARD_HEIGHT is None:
            BOARD_HEIGHT = len(self.data['players']['tiles']['board'][0])

    def GameOver(self):
        return self.data['gameOver']

    def Reward(self):
        return self.data['players']['reward']

    def GetPerception(self):
        perception = torch.zeros([BOARD_WIDTH, BOARD_HEIGHT, 5], dtype=torch.float)
        for x in range(BOARD_WIDTH):
            for y in range(BOARD_HEIGHT):
                tile = self.data['players']['tiles']['board'][x][y]

                if tile != 0:
                    perception[x][y][tile - 1] = 1

        x = self.data['players']['tiles']['falling']['x']
        y = self.data['players']['tiles']['falling']['y']
        top = self.data['players']['tiles']['falling']['top']
        bottom = self.data['players']['tiles']['falling']['bottom']
        perception[x][y + 0][top - 1] = 1
        perception[x][y + 1][bottom - 1] = 1

        perception[x][y + 0][4] = 1
        perception[x][y + 1][4] = 1

        return perception.permute([2, 0, 1]).to('cuda')


class Experience:
    def __init__(self, originalState: State, resultingState: State, action: int, reward: int):
        self.originalState = originalState
        self.resultingState = resultingState
        self.action = action
        self.reward = reward

    def OriginalState(self):
        return self.originalState

    def ResultingState(self):
        return self.resultingState

    def Action(self):
        return self.action

    def Reward(self):
        return self.reward


class ExperienceBuffer:
    def __init__(self):
        self.memory = deque(maxlen=EXPERIENCE_BUFFER_SIZE)

    def Push(self, experience: Experience):
        self.memory.append(experience)

    def Sample(self, count: int):
        return random.sample(self.memory, count)

    def Size(self):
        return len(self.memory)


# Model class
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.own_tiles_convolution = nn.Sequential(
            nn.Conv2d(5, 128, 7, padding='same'),
            nn.ReLU(),
            nn.Conv2d(128, 128, 7, padding='same'),
            nn.ReLU(),
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(128 * BOARD_WIDTH * BOARD_HEIGHT, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, ACTION_COUNT),
        )


    def forward(self, x):
        # Split input into our board and the enemy board
        our_board = self.own_tiles_convolution(x)

        combined = our_board.flatten()

        actions = self.linear_layers(combined)

        return actions


# Initialise model and experience buffer
MODEL_ID = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M")
CSV_FILE = open(f'./{MODEL_ID}.csv', "w", newline='')
CSV_WRITER = csv.writer(CSV_FILE)
CSV_WRITER.writerow(['EPISODE', "TURN COUNT", "REWARD", "REWARD RATE", "EPSILON"])
POLICY_NETWORKS = None
OPTIMIZERS      = None
EXPERIENCE_BUFFER = ExperienceBuffer()


def InitialiseModels():
    global POLICY_NETWORKS, OPTIMIZERS
    POLICY_NETWORKS = [Network().to("cuda"), Network().to("cuda")]
    OPTIMIZERS = [torch.optim.AdamW(POLICY_NETWORKS[0].parameters()), torch.optim.AdamW(POLICY_NETWORKS[1].parameters())]


# Connect to Puyo-Puyo server
SERVER = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
SERVER.connect(("127.0.0.1", 25500))
print("Connected to Server!")


def RandomActions():
    return random.randint(0, ACTION_COUNT - 1)


def SelectActions(state):
    if random.random() < EPSILON:
        return random.randint(0, ACTION_COUNT - 1)
        perception = state.GetPerception()

        output = POLICY_NETWORKS[0](perception)

        choice = int(output.max(0).indices.view(1, 1))

        return choice
    else:
        return RandomActions()


def OptimizeModel():
    # Skip if we don't have enough experiences to make a batch
    if EXPERIENCE_BUFFER.Size() < 5 * BATCH_SIZE:
        return

    selected_network = random.randint(0, 1)
    other_network = (selected_network + 1) % 2

    batch = EXPERIENCE_BUFFER.Sample(BATCH_SIZE)
    actions = torch.tensor([[s.Action()] for s in batch]).to('cuda')
    rewards = torch.tensor([[s.Reward()] for s in batch]).to('cuda')

    predicted_action_values = torch.stack([POLICY_NETWORKS[selected_network](s.OriginalState().GetPerception()) for s in batch]).gather(1, actions)

    torch.set_grad_enabled(False)
    next_state_values = torch.Tensor([[POLICY_NETWORKS[other_network](s.ResultingState().GetPerception()).max(0).values] if not s.ResultingState().GameOver() else [0] for s in batch]).to('cuda')
    torch.set_grad_enabled(True)
    target_action_values = rewards + DISCOUNT_FACTOR * next_state_values

    lossFunction = nn.HuberLoss()
    loss = lossFunction(predicted_action_values, target_action_values)

    OPTIMIZERS[selected_network].zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(POLICY_NETWORKS[selected_network].parameters(), 100)
    OPTIMIZERS[selected_network].step()


for i in range(EPISODE_COUNT):
    state = None
    previousState = None
    selected_action = None

    EPSILON = max(EXPLORATION_PROBABILITY - EXPLORATION_PROBABILITY * math.log(float(i) + 1.0, EPISODE_COUNT), 0)
    reward = 0
    turn_count = 0


    while True:
        previousState = state
        state = State(SERVER.recv(4096))
        reward += state.Reward()
        turn_count += 1

        if state is not None and previousState is not None and selected_action is not None:
            experience = Experience(previousState, state, selected_action, state.Reward())
            EXPERIENCE_BUFFER.Push(experience)

        OptimizeModel()

        if POLICY_NETWORKS is None:
            InitialiseModels()

        if state.GameOver():
            CSV_WRITER.writerow([i, turn_count] + [reward] + [reward / turn_count, EPSILON])
            CSV_FILE.flush()
            break

        selected_action = SelectActions(state)
        SERVER.send(bytearray([selected_action]))


torch.save(POLICY_NETWORKS[0].state_dict(), f"{MODEL_ID}.pt")