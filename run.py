import re
from pathlib import Path

import pygame
import torch

import src.ai.model as model
import src.tetris.game as game


MODELS_DIR = Path("models/")

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

FPS = 60
AI_STEP_DELAY_MS = 75


def get_available_models():
    if not MODELS_DIR.exists():
        return []

    return sorted(
        directory.name
        for directory in MODELS_DIR.iterdir()
        if directory.is_dir()
    )


def get_model_name():
    available_models = get_available_models()

    if not available_models:
        raise FileNotFoundError(
            f"No models found inside '{MODELS_DIR}'."
        )

    print("\nAvailable models:")

    for name in available_models:
        print(f"  - {name}")

    print()

    while True:
        name = input("Enter model name: ").strip()

        if name in available_models:
            return name

        print(f"Model '{name}' not found.\n")


def get_available_episodes(model_path):
    episodes = []

    pattern = re.compile(
        r"checkpoint_(\d+)\.pth$"
    )

    for file in model_path.iterdir():
        if not file.is_file():
            continue

        match = pattern.fullmatch(file.name)

        if match:
            episodes.append(
                int(match.group(1))
            )

    return sorted(episodes)


def get_episode(model_path):
    episodes = get_available_episodes(
        model_path
    )

    if not episodes:
        raise FileNotFoundError(
            f"No checkpoints found in '{model_path}'."
        )

    print(
        f"\nAvailable checkpoints: "
        f"{episodes[0]} - {episodes[-1]}"
    )

    print(
        "Press Enter to use the latest checkpoint."
    )

    while True:
        user_input = input(
            "Enter episode: "
        ).strip()

        if user_input == "":
            return episodes[-1]

        try:
            episode = int(user_input)

        except ValueError:
            print(
                "Please enter an episode number."
            )
            continue

        if episode in episodes:
            return episode

        print(
            f"Checkpoint {episode} does not exist."
        )


def load_model(model_name, episode):
    checkpoint_path = (
        MODELS_DIR
        / model_name
        / f"checkpoint_{episode}.pth"
    )

    print(
        f"\nLoading: {checkpoint_path}"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE
    )

    ai_model = model.TetrisAI().to(DEVICE)

    ai_model.load_state_dict(
        checkpoint["model_state"]
    )

    ai_model.eval()

    print(
        f"Loaded '{model_name}' "
        f"episode {episode}"
    )

    return ai_model


def state_to_tensor(state):
    return torch.tensor(
        state,
        dtype=torch.float32,
        device=DEVICE
    ).unsqueeze(0)


def choose_action(ai_model, state):
    state_tensor = state_to_tensor(state)

    with torch.no_grad():
        q_values = ai_model(
            state_tensor
        )

    action = q_values.argmax(
        dim=1
    ).item()

    return action


def run_ai(ai_model, model_name, episode):
    tetris = game.Tetris()

    state = tetris.reset()

    pygame.init()
    screen = pygame.display.set_mode(
        (
            game.SCREEN_WIDTH,
            game.SCREEN_HEIGHT
        )
    )

    pygame.display.set_caption(
        f"Tetr.AI - {model_name} - Episode {episode}"
    )

    clock = pygame.time.Clock()

    running = True
    paused = False

    last_ai_step = pygame.time.get_ticks()

    game_number = 1
    rows_cleared = 0
    steps = 0

    action_names = [
        "Nothing",
        "Left",
        "Right",
        "Rotate",
        "Hard Drop"
    ]

    current_action = "Waiting"

    while running:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_SPACE:
                    paused = not paused

        current_time = pygame.time.get_ticks()

        if (
            not paused
            and current_time - last_ai_step
            >= AI_STEP_DELAY_MS
        ):
            action = choose_action(
                ai_model,
                state
            )

            current_action = action_names[action]

            state, done, cleared_rows = (
                tetris.step(action)
            )

            rows_cleared += cleared_rows
            steps += 1

            last_ai_step = current_time

            if done:
                print(
                    f"Game {game_number} | "
                    f"Rows: {rows_cleared} | "
                    f"Steps: {steps}"
                )

                game_number += 1
                rows_cleared = 0
                steps = 0

                state = tetris.reset()

        screen.fill(game.BLACK)

        game.draw_grid(
            screen,
            tetris.grid,
            tetris.current_piece
        )

        pygame.display.set_caption(
            f"Tetr.AI | "
            f"{model_name} | "
            f"Episode {episode} | "
            f"Game {game_number} | "
            f"Rows {rows_cleared} | "
            f"Action: {current_action}"
        )

        pygame.display.flip()

    pygame.quit()


def main():
    print(
        f"Using device: {DEVICE}"
    )

    model_name = get_model_name()

    model_path = (
        MODELS_DIR / model_name
    )

    episode = get_episode(
        model_path
    )

    ai_model = load_model(
        model_name,
        episode
    )

    run_ai(
        ai_model,
        model_name,
        episode
    )


if __name__ == "__main__":
    main() 