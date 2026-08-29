from multiprocessing import Process
import tetris.game as game

AMOUNT_OF_PROCESSES = 1

def start_game():
    tetris = game.Tetris()
    tetris.start()


if __name__ == "__main__":
    processes = []

    for _ in range(AMOUNT_OF_PROCESSES):
        p = Process(target=start_game)
        p.start()
        processes.append(p)

    for p in processes:
        p.join()