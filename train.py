# RUN THIS SCRIPT TO TRAIN YOUR MODEL
import src.ai.trainer as trainer
import src.tetris.game as game
import threading

Tetris = game.Tetris()
model = trainer.Trainer(Tetris)

print(f"Device: {model.device} | Starting epsilon: {model.epsilon} | Gamma: {model.gamma}")
name = input("What is the name of the AI you want to train > ")

model.train(name)