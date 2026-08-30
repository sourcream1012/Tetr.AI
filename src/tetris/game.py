import random

import pygame

from . import piece


pygame.init()

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 700
PLAY_WIDTH = 300
PLAY_HEIGHT = 600
CELL_SIZE = 30

BOARD_WIDTH = 10
BOARD_HEIGHT = 20

OFFSET_X = (SCREEN_WIDTH - PLAY_WIDTH) // 2
OFFSET_Y = (SCREEN_HEIGHT - PLAY_HEIGHT) // 2

BLACK = (0, 0, 0)
GRID_COLOR = (40, 40, 40)


def create_grid(locked_positions=None):
    if locked_positions is None:
        locked_positions = {}

    grid = [
        [BLACK for _ in range(BOARD_WIDTH)]
        for _ in range(BOARD_HEIGHT)
    ]

    for y in range(BOARD_HEIGHT):
        for x in range(BOARD_WIDTH):
            if (x, y) in locked_positions:
                grid[y][x] = locked_positions[(x, y)]

    return grid


def check_empty_cells(grid):
    empty_cells = []

    for y in range(len(grid)):
        for x in range(len(grid[y])):
            if grid[y][x] == BLACK:
                empty_cells.append((x, y))

    return empty_cells


def render_shapes(current_piece, grid):
    empty_cells = check_empty_cells(grid)

    for x, y in piece.get_piece_positions(current_piece):
        if grid[y][x] in empty_cells:
            grid[y][x] = BLACK
        else:
            grid[y][x] = current_piece.color


def draw_grid(screen, grid, current_piece):
    for row in range(len(grid)):
        for col in range(len(grid[row])):
            x = OFFSET_X + col * CELL_SIZE
            y = (OFFSET_Y + 25) + row * CELL_SIZE

            if grid[row][col] != BLACK:
                pygame.draw.rect(
                    screen,
                    grid[row][col],
                    (x, y, CELL_SIZE, CELL_SIZE),
                )
            else:
                pygame.draw.rect(
                    screen,
                    GRID_COLOR,
                    (x, y, CELL_SIZE, CELL_SIZE),
                    1,
                )

    for col, row in piece.get_piece_positions(current_piece):
        x = OFFSET_X + col * CELL_SIZE
        y = (OFFSET_Y + 25) + row * CELL_SIZE

        pygame.draw.rect(
            screen,
            current_piece.color,
            (x, y, CELL_SIZE, CELL_SIZE),
        )


def lock_piece(current_piece, locked_positions):
    for x, y in piece.get_piece_positions(current_piece):
        if y >= 0:
            locked_positions[(x, y)] = current_piece.color


def is_valid_space(current_piece, grid):
    positions = piece.get_piece_positions(current_piece)

    for x, y in positions:
        if x < 0 or x >= BOARD_WIDTH or y >= BOARD_HEIGHT:
            return False

        if y < 0:
            continue

        if grid[y][x] != BLACK:
            return False

    return True


def delete_rows(grid, locked_positions):
    completed_rows = [
        y
        for y in range(len(grid))
        if all(cell != BLACK for cell in grid[y])
    ]

    if not completed_rows:
        return 0

    for y in completed_rows:
        for x in range(BOARD_WIDTH):
            if (x, y) in locked_positions:
                del locked_positions[(x, y)]

    new_locked_positions = {}

    for (x, y), color in locked_positions.items():
        rows_below = sum(
            1 for completed_y in completed_rows if completed_y > y
        )

        new_y = y + rows_below
        new_locked_positions[(x, new_y)] = color

    locked_positions.clear()
    locked_positions.update(new_locked_positions)

    return len(completed_rows)


def get_piece():
    shape = random.choice(piece.shapes)

    cells = piece.get_shape_cells(shape, 0)
    shape_width = max(x for x, y in cells) + 1

    spawn_x = (BOARD_WIDTH - shape_width) // 2
    spawn_y = 0

    return piece.Piece(spawn_x, spawn_y, shape)

def print_grid(grid, current_piece):
    # FOR TESTING PURPOSES
    display_grid = [row[:] for row in grid]

    for x, y in piece.get_piece_positions(current_piece):
        if 0 <= y < BOARD_HEIGHT and 0 <= x < BOARD_WIDTH:
            display_grid[y][x] = current_piece.color

    for row in display_grid:
        print(
            " ".join(
                "." if cell == BLACK else "#"
                for cell in row
            )
        )

class Tetris:
    def __init__(self):
        self.locked_positions = {}
        self.grid = create_grid(self.locked_positions)
        self.current_piece = get_piece()
        self.next_piece = get_piece()
        self.score = 0

    def reset(self):
        self.locked_positions = {}
        self.grid = create_grid(self.locked_positions)
        self.current_piece = get_piece()
        self.next_piece = get_piece()
        self.score = 0

    def format_ai_readable(self):
        new_grid = [[0 for _ in range(10)] for _ in range(20)]

        for row in self.grid:
            for cell in self.grid:
                if cell != (0, 0, 0):
                    new_grid[row][cell] = 1

    def move_left(self):
        self.current_piece.x -= 1
        
        if not is_valid_space(
            self.current_piece,
            self.grid,
        ):
            self.current_piece.x += 1

    def move_right(self):
        self.current_piece.x += 1
                
        if not is_valid_space(
            self.current_piece,
            self.grid,
        ):
            self.current_piece.x -= 1

    def rotate(self):
        self.current_piece.rotation += 1
                
        if not is_valid_space(
            self.current_piece,
            self.grid,
        ):
            self.current_piece.rotation -= 1

    def soft_drop(self):
        self.current_piece.y += 1
        
        if not is_valid_space(
            self.current_piece,
            self.grid,
        ):
            self.current_piece.y -= 1

    def hard_drop(self):
        while is_valid_space(
            self.current_piece,
            self.grid,
        ):
            self.current_piece.y += 1
        
        self.current_piece.y -= 1

    def start_ai_env(self):
        """Starts a Tetris game without rendering with Pygame. Used for training the AI."""
        clock = pygame.time.Clock()
        fall_timer = 0
        fall_speed = 0.5

        while True:
            dt = clock.tick(60) / 1000
            fall_timer += dt

            if fall_timer >= fall_speed:
                self.current_piece.y += 1

                if not is_valid_space(
                    self.current_piece,
                    self.grid,
                ):
                    self.current_piece.y -= 1

                    lock_piece(
                        self.current_piece,
                        self.locked_positions,
                    )

                    self.grid = create_grid(self.locked_positions)

                    delete_rows(
                        self.grid,
                        self.locked_positions,
                    )

                    self.grid = create_grid(self.locked_positions)

                    self.current_piece = self.next_piece
                    self.next_piece = get_piece()

                    if not is_valid_space(
                        self.current_piece,
                        self.grid,
                    ):
                        self.reset()
                        fall_timer = 0

                fall_timer = 0

                print_grid(self.grid, self.current_piece)
                print()  
        
    def start(self):
        """Starts a Tetris game while rendering with Pygame."""
        clock = pygame.time.Clock()
        screen = pygame.display.set_mode(
            (SCREEN_WIDTH, SCREEN_HEIGHT)
        )

        running = True
        fall_timer = 0
        fall_speed = 0.5

        while running:
            dt = clock.tick(60) / 1000

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        self.current_piece.x -= 1

                        if not is_valid_space(
                            self.current_piece,
                            self.grid,
                        ):
                            self.current_piece.x += 1

                    elif event.key == pygame.K_RIGHT:
                        self.current_piece.x += 1

                        if not is_valid_space(
                            self.current_piece,
                            self.grid,
                        ):
                            self.current_piece.x -= 1

                    elif event.key == pygame.K_SPACE:
                        while is_valid_space(
                            self.current_piece,
                            self.grid,
                        ):
                            self.current_piece.y += 1

                        self.current_piece.y -= 1

                        lock_piece(
                            self.current_piece,
                            self.locked_positions,
                        )

                        self.grid = create_grid(self.locked_positions)
                        delete_rows(
                            self.grid,
                            self.locked_positions,
                        )
                        self.grid = create_grid(self.locked_positions)

                        self.current_piece = self.next_piece
                        self.next_piece = get_piece()

                        if not is_valid_space(
                            self.current_piece,
                            self.grid,
                        ):
                            self.reset()
                            fall_timer = 0

                    elif event.key == pygame.K_UP:
                        self.current_piece.rotation += 1

                        if not is_valid_space(
                            self.current_piece,
                            self.grid,
                        ):
                            self.current_piece.rotation -= 1

            keys = pygame.key.get_pressed()

            if keys[pygame.K_DOWN]:
                self.current_piece.y += 1

                if not is_valid_space(
                    self.current_piece,
                    self.grid,
                ):
                    self.current_piece.y -= 1

            fall_timer += dt

            screen.fill(BLACK)
            draw_grid(
                screen,
                self.grid,
                self.current_piece,
            )

            if fall_timer >= fall_speed:
                self.current_piece.y += 1

                if not is_valid_space(
                    self.current_piece,
                    self.grid,
                ):
                    self.current_piece.y -= 1

                    lock_piece(
                        self.current_piece,
                        self.locked_positions,
                    )

                    self.grid = create_grid(self.locked_positions)
                    delete_rows(
                        self.grid,
                        self.locked_positions,
                    )
                    self.grid = create_grid(self.locked_positions)

                    self.current_piece = self.next_piece
                    self.next_piece = get_piece()

                    if not is_valid_space(
                        self.current_piece,
                        self.grid,
                    ):
                        self.reset()
                        fall_timer = 0

                fall_timer = 0

            pygame.display.flip()

        pygame.quit()