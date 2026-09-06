import random

from . import model

import torch
import torch.nn as nn

class Trainer:
    def __init__(
        self, 
        tetris,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=0.1,
    ):
        self.Tetris = tetris

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.TetrisAI().to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=learning_rate
        )

        self.loss_fn = nn.SmoothL1Loss()

        self.gamma = gamma
        self.epsilon = epsilon

    def state_to_tensor(self, state):
        return torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(5)

        state_tensor = self.state_to_tensor(state)
        
        with torch.no_grad():
            q_values = self.model(state_tensor)

        return q_values.argmax(dim=1).item()

    def calculate_reward(self, cleared_rows, done):
        reward = 0

        if cleared_rows == 1:
            reward += 10
        elif cleared_rows == 2:
            reward += 30
        elif cleared_rows == 3:
            reward += 60
        elif cleared_rows == 4:
            reward += 100

        if done:
            reward -= 100

        return reward

    def train(self):
        while True:
            pass