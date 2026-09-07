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

        self.memory = []

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
            reward += 15
        elif cleared_rows == 3:
            reward += 20
        elif cleared_rows == 4:
            reward += 30

        if done:
            reward -= 25

        return reward

    def remember(self, state, action, reward, next_state, done):
        self.memory.append(
            (state, action, reward, next_state, done)
        )

    def train_step(self, batch_size=64):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(
            self.memory,
            batch_size
        )

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for state, action, reward, next_state, done in batch:
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        states = torch.tensor(
            states,
            dtype=torch.float32,
            device=self.device
        )

        actions = torch.tensor(
            actions,
            dtype=torch.long,
            device=self.device
        )

        rewards = torch.tensor(
            rewards,
            dtype=torch.float32,
            device=self.device
        )

        next_states = torch.tensor(
            next_states,
            dtype=torch.float32,
            device=self.device
        )

        dones = torch.tensor(
            dones,
            dtype=torch.bool,
            device=self.device
        )

        current_q_values = self.model(states)

        current_q = current_q_values.gather(
            1,
            actions.unsqueeze(1)
        ).squeeze(1)

        with torch.no_grad():
            next_q_values = self.model(next_states)

            max_next_q = next_q_values.max(dim=1).values

            target_q = rewards + (self.gamma * max_next_q * (~dones))

        loss = self.loss_fn(
            current_q,
            target_q
        )

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        return (
            loss.item(),
            current_q.mean().item(),
            target_q.mean().item()
        )

    def train(self):
        state = self.Tetris.reset()

        episode = 1
        episode_reward = 0
        episode_rows = 0
        episode_steps = 0

        while True:
            action = self.choose_action(state)

            next_state, done, cleared_rows = self.Tetris.step(action)

            reward = self.calculate_reward(
                cleared_rows,
                done
            )

            self.remember(
                state,
                action,
                reward,
                next_state,
                done
            )

            training_data = self.train_step()

            episode_reward += reward
            episode_rows += cleared_rows
            episode_steps += 1

            state = next_state

            if done:
                if training_data is not None:
                    loss, avg_q, avg_target_q = training_data

                    print(
                        f"Episode: {episode} | "
                        f"Reward: {episode_reward} | "
                        f"Rows: {episode_rows} | "
                        f"Steps: {episode_steps} | "
                        f"Loss: {loss:.3f} | "
                        f"Avg Q: {avg_q:.3f} | "
                        f"Target Q: {avg_target_q:.3f} | "
                        f"Epsilon: {self.epsilon:.3f} | "
                        f"Memory: {len(self.memory)}"
                    )

                episode += 1

                episode_reward = 0
                episode_rows = 0
                episode_steps = 0

                state = self.Tetris.reset()   