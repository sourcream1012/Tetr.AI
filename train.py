# RUN THIS SCRIPT TO TRAIN YOUR MODEL

import json
import sys
from pathlib import Path

import src.ai.trainer as trainer
import src.tetris.game as game


def get_base_directory():
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent

    return Path(__file__).resolve().parent


BASE_DIR = get_base_directory()
CONFIG_PATH = BASE_DIR / "training_config.json"


with open(CONFIG_PATH, "r") as file:
    config = json.load(file)


Tetris = game.Tetris()

model = trainer.Trainer(
    Tetris,
    config=config
)

print(
    f"Device: {model.device} | "
    f"Starting epsilon: {model.epsilon} | "
    f"Gamma: {model.gamma}"
)

name = input("What is the name of the AI you want to train > ")

model.train(name)