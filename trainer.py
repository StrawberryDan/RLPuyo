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

EPISODE_COUNT = 500
BATCH_SIZE = 16
BOARD_WIDTH = None
BOARD_HEIGHT = None
ACTION_COUNT = 6
EXPERIENCE_BUFFER_SIZE = 10000
LEARNING_RATE = 1e-4
DISCOUNT_FACTOR = 0.9
EXPLORATION_PROBABILITY = 0.8
EPSILON = EXPLORATION_PROBABILITY


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

    def Reward(self, playerIndex):
        return self.data['players'][playerIndex]['reward']

    def GetPerception(self, playerIndex):
        perception = torch.zeros([BOARD_WIDTH, BOARD_HEIGHT, 5], dtype=torch.float)
        for x in range(BOARD_WIDTH):
            for y in range(BOARD_HEIGHT):
                tile = self.data['players'][playerIndex]['tiles']['board'][x][y]

                if tile != 0:
                    perception[x][y][tile - 1] = 1

        x = self.data['players'][playerIndex]['tiles']['falling']['x']
        y = self.data['players'][playerIndex]['tiles']['falling']['y']
        top = self.data['players'][playerIndex]['tiles']['falling']['top']
        bottom = self.data['players'][playerIndex]['tiles']['falling']['bottom']
        perception[x][y + 0][top - 1] = 1
        perception[x][y + 1][bottom - 1] = 1

        perception[x][y + 0][4] = 1
        perception[x][y + 1][4] = 1

        return perception


class Experience:
    def __init__(self, originalState: State, resultingState: State, actions: list[int], rewards: list[int]):
        self.originalState = originalState
        self.resultingState = resultingState
        self.action = actions
        self.reward = rewards

    def OriginalState(self):
        return self.originalState

    def ResultingState(self):
        return self.resultingState

    def Action(self, playerIndex):
        return self.action[playerIndex]

    def Reward(self, playerIndex):
        return self.reward[playerIndex]


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
            nn.Conv2d(5, 64, 7, padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 64, 7, padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 64, 7, padding='same'),
            nn.ReLU(),
        )

        self.enemy_tiles_convolution = nn.Sequential(
            nn.Conv2d(5, 64, 7, padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 64, 7, padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 64, 7, padding='same'),
            nn.ReLU(),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(2 * 64 * BOARD_WIDTH * BOARD_HEIGHT, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, ACTION_COUNT),
            nn.ReLU(),
            nn.Softmax(dim=0),
        )

    def forward(self, x):
        # Split input into our board and the enemy board
        our_board, their_board = x.split([5 * BOARD_WIDTH * BOARD_HEIGHT, 5 * BOARD_WIDTH * BOARD_HEIGHT])

        our_board = our_board.reshape((BOARD_WIDTH, BOARD_HEIGHT, 5))
        our_board = our_board.permute((2, 0, 1))
        our_board = self.own_tiles_convolution(our_board.cuda())

        their_board = their_board.reshape((BOARD_WIDTH, BOARD_HEIGHT, 5))
        their_board = their_board.permute((2, 0, 1))
        their_board = self.enemy_tiles_convolution(their_board.cuda())

        combined = torch.cat([our_board.flatten(), their_board.flatten()])

        actions = self.linear_layers(combined)

        return actions


# Initialise model and experience buffer
MODEL_ID = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M")
CSV_FILE = open(f'./{MODEL_ID}.csv', "w", newline='')
CSV_WRITER = csv.writer(CSV_FILE)
CSV_WRITER.writerow(['EPISODE', "TURN COUNT", "REWARDS 0", "REWARDS 1", "REWARD RATE 0", "REWARD RATE 1", "EPSILON"])
POLICY_NETWORK = None
TARGET_NETWORK = None
OPTIMIZER      = None
EXPERIENCE_BUFFER = ExperienceBuffer()


def InitialiseModels():
    global POLICY_NETWORK, TARGET_NETWORK, OPTIMIZER
    POLICY_NETWORK = Network().to("cuda")
    TARGET_NETWORK = Network().to("cuda")
    TARGET_NETWORK.load_state_dict(POLICY_NETWORK.state_dict())
    OPTIMIZER = torch.optim.AdamW(POLICY_NETWORK.parameters(), lr=LEARNING_RATE, amsgrad=True)


# Connect to Puyo-Puyo server
SERVER = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
SERVER.connect(("127.0.0.1", 25500))
print("Connected to Server!")


def RandomActions():
    return [random.randint(0, ACTION_COUNT - 1), random.randint(0, ACTION_COUNT - 1)]


def SelectActions(state):
    if random.random() < EPSILON:
        return [random.randint(0, ACTION_COUNT - 1), random.randint(0, ACTION_COUNT - 1)]
    else:
        perception = [
            torch.cat([torch.flatten(state.GetPerception(0)), torch.flatten(state.GetPerception(1))]),
            torch.cat([torch.flatten(state.GetPerception(1)), torch.flatten(state.GetPerception(0))])
        ]

        outputs = [POLICY_NETWORK(p) for p in perception]

        choices = [int(o.max(0).indices.view(1, 1)) for o in outputs]

        return choices


def OptimizeModel():
    # Skip if we don't have enough experiences to make a batch
    if EXPERIENCE_BUFFER.Size() < BATCH_SIZE:
        return


    batch = EXPERIENCE_BUFFER.Sample(BATCH_SIZE)


    for player in range(2):
        states  = [torch.cat([s.OriginalState().GetPerception((player + 0) % 2).flatten(), s.OriginalState().GetPerception((player + 1) % 2).flatten()]) for s in batch]
        next_states = [torch.cat([s.ResultingState().GetPerception((player + 0) % 2).flatten(), s.ResultingState().GetPerception((player + 1) % 2).flatten()]) for s in batch]
        actions = torch.tensor([[s.Action(player)] for s in batch]).to('cuda')
        rewards = torch.tensor([[s.Reward(player)] for s in batch]).to('cuda')


        predicted_action_values = torch.stack([POLICY_NETWORK(s) for s in states]).gather(1, actions)


        next_state_values = torch.Tensor([[TARGET_NETWORK(s).max(0).values] for s in next_states]).to('cuda')
        target_action_values = rewards + DISCOUNT_FACTOR * next_state_values

        lossFunction  = nn.HuberLoss()
        loss = lossFunction(predicted_action_values, target_action_values)

        loss.backward()
        OPTIMIZER.step()
        OPTIMIZER.zero_grad()


for i in range(EPISODE_COUNT):
    state = None
    previousState = None
    selected_actions = None

    EPSILON = max(EXPLORATION_PROBABILITY - EXPLORATION_PROBABILITY * math.log(float(i) + 1.0, EPISODE_COUNT), 0)
    rewards = [0, 0]
    turn_count = 0


    while True:
        previousState = state
        state = State(SERVER.recv(4096))
        rewards[0] += state.Reward(0)
        rewards[1] += state.Reward(1)
        turn_count += 1

        if state is not None and previousState is not None and selected_actions is not None:
            experience = Experience(previousState, state, selected_actions, [state.Reward(0), state.Reward(1)])
            EXPERIENCE_BUFFER.Push(experience)

        OptimizeModel()

        if POLICY_NETWORK is None:
            InitialiseModels()

        if state.GameOver():
            CSV_WRITER.writerow([i, turn_count] + rewards + [rewards[0] / turn_count, rewards[1] / turn_count, EPSILON])
            CSV_FILE.flush()
            TARGET_NETWORK.load_state_dict(POLICY_NETWORK.state_dict())
            break

        selected_actions = SelectActions(state)
        SERVER.send(bytearray(selected_actions))


torch.save(POLICY_NETWORK.state_dict(), f"{MODEL_ID}.pt")