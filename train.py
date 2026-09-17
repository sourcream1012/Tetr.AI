# RUN THIS SCRIPT TO TRAIN YOUR MODEL
import src.ai.trainer as trainer
import src.tetris.game as game
import threading

name = input("What is the name of the AI you want to train > ")

Tetris = game.Tetris()
model = trainer.Trainer(Tetris)

model.train(name)