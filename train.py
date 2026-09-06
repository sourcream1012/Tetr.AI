# RUN THIS SCRIPT TO TRAIN YOUR MODEL
import src.ai.trainer as trainer
import src.tetris.game as game
import threading

Tetris = game.Tetris()
AI_Trainer = trainer.Trainer(Tetris)

def start_env():
    Tetris.start_ai_env(True, False, 0)

thread_start_env = threading.Thread(target=start_env)
thread_start_env.start()

# CORE TRAINING LOOP
while True:
    pass
    
