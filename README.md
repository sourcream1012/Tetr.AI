# Tetr.AI
A neural network made to play the popular russian game Tetris. Made for the Hack Club Stardance YSWS.

## Getting started

### Windows setup

On windows, run:

- **TetrAI-TrainAI.exe** to train an AI. If you want to change training values change them in training_config.json
- **TetrAI-RunAI.exe** to run an AI model.
- **Tetr.AI-PlayTetris.exe** to play Tetris for fun.
- **Tetr.AI-Graph.exe** to graph a models statistics from their training logs.

### Linux

Unfortunatly I am not sure if the executable files work on linux.

## Project status
Tetr.AI is currently a fully functional Tetris environment where a neural network can play Tetris.

The training pipeline, checkpoint system, model runner, game environment, and logging system are all functional.

The included trained models are purely experimental. They have learned how to survive but they can't clear rows. Due to the time it takes to train reinforcement learning models, the current release will not contain fully trained models capable of playing Tetris well.

I will continue to develop for the time being following my first release and my ship to Stardance. The development would particularly be around reward shaping and solving my current sparse-reward problem.

## Model History

### Tetr.AI-V1 - Failed experiment
My first long duration training attempt

**Result:** model learned some survival behavior but did not learn to reliably clear rows

**What I learned:** 
1. Epsilon decayed too slowly decaying at a rate of 0.999
2. Punishment for holes is too severe punishing every new hole made by -10
```python
reward += (new_holes - old_holes) * -10
```
[Tetr.AI-V1 Weights](models/Tetr.AI-V1)

[Tetr.AI-V1 training logs](old_training_logs/Tetr.AI-V1/training.csv)

### Tetr.AI-V2 - Failed experiment
Second large training attempt lasting roughly the same amount of time as Tetr.AI-V1

**Changes from V1:**
- Holes penalty from -10 per new hole created to -5 per new hole created
- Epsilon decay from 0.999 to 0.9999

**Result:** Model improved compared to Tetr.AI-V1 but very slightly, model still doesn't learn the actual objective of Tetris which is to clear rows

**What I learned:**
1. Reward shaping still needs work particularly in the clearing rows
2. Holes penalty is still too harsh (reduced it to -5 per hole yet clearing rows still does not compete)
3. It is rare for the AI to actually clear rows randomly leading to a sparse-reward problem

**My hypothesis:** Because it is rare for the AI to clear rows randomly and the reward still seems low I will need to change these values
- Holes penalty possibly to -2 per new hole created
- Increase cleared rows reward by 1.5x-2x
- Make the batch size in the Trainer.train_step() function auto bias cleared rows to make the AI realize "I need to be learning more off of the instances where I actually cleared rows" which would hopefully partially solve the sparse-reward problem

[Tetr.AI-V2 Weights](models/Tetr.AI-V2)

[Tetr.AI-V2 training logs](old_training_logs/Tetr.AI-V2/training.csv)
