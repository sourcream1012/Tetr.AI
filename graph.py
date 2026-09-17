import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


LOG_FILE = Path("training_logs") / "training.csv"

ROLLING_WINDOW = 100


def main():
    data = pd.read_csv(LOG_FILE)

    # Create rolling averages
    data["reward_avg"] = (
        data["reward"]
        .rolling(ROLLING_WINDOW)
        .mean()
    )

    data["rows_avg"] = (
        data["rows"]
        .rolling(ROLLING_WINDOW)
        .mean()
    )

    data["steps_avg"] = (
        data["steps"]
        .rolling(ROLLING_WINDOW)
        .mean()
    )

    plt.figure()

    plt.plot(
        data["episode"],
        data["reward"],
        alpha=0.2,
        label="Episode Reward"
    )

    plt.plot(
        data["episode"],
        data["reward_avg"],
        label=f"{ROLLING_WINDOW}-Episode Average"
    )

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Reward")
    plt.legend()
    plt.grid()

    plt.show()

    plt.figure()

    plt.plot(
        data["episode"],
        data["rows"],
        alpha=0.2,
        label="Rows Cleared"
    )

    plt.plot(
        data["episode"],
        data["rows_avg"],
        label=f"{ROLLING_WINDOW}-Episode Average"
    )

    plt.xlabel("Episode")
    plt.ylabel("Rows Cleared")
    plt.title("Rows Cleared Per Episode")
    plt.legend()
    plt.grid()

    plt.show()

    plt.figure()

    plt.plot(
        data["episode"],
        data["steps_avg"]
    )

    plt.xlabel("Episode")
    plt.ylabel("Average Steps")
    plt.title(
        f"Episode Length ({ROLLING_WINDOW}-Episode Average)"
    )

    plt.grid()

    plt.show()

    plt.figure()

    plt.plot(
        data["episode"],
        data["avg_q"],
        label="Average Q"
    )

    plt.plot(
        data["episode"],
        data["target_q"],
        label="Target Q"
    )

    plt.xlabel("Episode")
    plt.ylabel("Q Value")
    plt.title("Predicted Q vs Target Q")
    plt.legend()
    plt.grid()

    plt.show()

    data["loss_avg"] = (
        data["loss"]
        .rolling(ROLLING_WINDOW)
        .mean()
    )

    plt.figure()

    plt.plot(
        data["episode"],
        data["loss"],
        alpha=0.2,
        label="Loss"
    )

    plt.plot(
        data["episode"],
        data["loss_avg"],
        label=f"{ROLLING_WINDOW}-Episode Average"
    )

    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid()

    plt.show()


if __name__ == "__main__":
    main()