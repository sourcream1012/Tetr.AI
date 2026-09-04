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
        self.fall_timer = 0
        self.fall_speed = 0.75

    def reset(self):
        self.locked_positions = {}
        self.grid = create_grid(self.locked_positions)
        self.current_piece = get_piece()
        self.next_piece = get_piece()
        self.score = 0
        self.fall_timer = 0
        self.fall_speed = 0.75

    def format_ai_readable(self):
        """Formats the current game state into a 2D array that can be used as the inputs for an AI model."""
        new_grid = [[0 for _ in range(BOARD_WIDTH)] for _ in range(BOARD_HEIGHT)]

        # Locked pieces
        for row in range(BOARD_HEIGHT):
            for cell in range(BOARD_WIDTH):
                if self.grid[row][cell] != BLACK:
                    new_grid[row][cell] = 1

        # Current falling piece
        for x, y in piece.get_piece_positions(self.current_piece):
            if 0 <= x < BOARD_WIDTH and 0 <= y < BOARD_HEIGHT:
                new_grid[y][x] = 1

        return new_grid

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
            self.fall_timer = 0

    def fall(self):
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

        self.fall_timer = 0

    def step(self, action):
        if action == 0:
            pass
        elif action == 1:
            self.move_left()
        elif action == 2:
            self.move_right()
        elif action == 3:
            self.rotate()
        elif action == 4:
            self.hard_drop()
        else:
            raise ValueError(f"Invalid action: {action}")

        return self.format_ai_readable() 

    def start_ai_env(self, render=False):
        """Starts the Tetris environment for AI training."""
        clock = pygame.time.Clock()
        running = True

        if render:
            screen = pygame.display.set_mode(
                (SCREEN_WIDTH, SCREEN_HEIGHT)
            )

        while running:
            dt = clock.tick(60) / 1000
            self.fall_timer += dt

            if render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False

                if self.fall_timer >= self.fall_speed:
                    self.fall()

                if render:
                    screen.fill(BLACK)

                    draw_grid(
                        screen,
                        self.grid,
                        self.current_piece,
                    )

                    pygame.display.flip()

                else:
                    print_grid(
                        self.grid,
                        self.current_piece,
                    )
                    print()

        if render:
            pygame.quit()  
        
    def start(self):
        """Starts a Tetris game while rendering with Pygame."""
        clock = pygame.time.Clock()
        screen = pygame.display.set_mode(
            (SCREEN_WIDTH, SCREEN_HEIGHT)
        )

        running = True

        while running:
            dt = clock.tick(60) / 1000

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        self.move_left()

                    elif event.key == pygame.K_RIGHT:
                        self.move_right()

                    elif event.key == pygame.K_SPACE:
                        self.hard_drop()

                    elif event.key == pygame.K_UP:
                        self.rotate()

            keys = pygame.key.get_pressed()

            if keys[pygame.K_DOWN]:
                self.soft_drop()

            self.fall_timer += dt

            screen.fill(BLACK)
            draw_grid(
                screen,
                self.grid,
                self.current_piece,
            )

            if self.fall_timer >= self.fall_speed:
                self.fall()
                self.fall_timer = 0

            pygame.display.flip()

        pygame.quit()