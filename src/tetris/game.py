import pygame
import random
from . import piece

pygame.init()

screenWidth = 800
screenHeight = 700
playWidth = 300
playHeight = 600
cellSize = 30

offset_x = (screenWidth - playWidth) // 2
offset_y = (screenHeight - playHeight) // 2 

def createGrid(locked_pos = {}):
    grid = [[(0, 0, 0) for _ in range(10)] for _ in range(20)]

    for y in range(len(grid)):
        for x in range(len(grid[y])):
            if (x, y) in locked_pos:
                c = locked_pos[(x, y)]
                grid[y][x] = c

    return grid

def checkEmptyCells(grid):
    emptyCells = []

    for y in range(len(grid)):
        for x in range(len(grid[y])):
            if grid[y][x] == (0, 0, 0):
                emptyCells.append((x, y))

    return emptyCells

def renderShapes(currentPiece, grid):
    for x, y in piece.getPiecePositions(currentPiece):
        if grid[y][x] in checkEmptyCells(grid):
            grid[y][x] = (0, 0, 0)
        else:
            grid[y][x] = currentPiece.color

def drawGrid(screen, grid, currentPiece):
    for row in range(len(grid)):
        for col in range(len(grid[row])):
            x = offset_x + col * cellSize
            y = (offset_y + 25) + row * cellSize

            if grid[row][col] != (0, 0, 0):
                pygame.draw.rect(
                    screen,
                    grid[row][col],
                    (x, y, cellSize, cellSize)
                )
            else:
                pygame.draw.rect(
                    screen,
                    (40, 40, 40),
                    (x, y, cellSize, cellSize),
                    1
                )

    for col, row in piece.getPiecePositions(currentPiece):
        x = offset_x + col * cellSize
        y = (offset_y + 25) + row * cellSize

        pygame.draw.rect(
            screen,
            currentPiece.color,
            (x, y, cellSize, cellSize)
        )

def lockPiece(currentPiece, locked_pos):
    for x, y in piece.getPiecePositions(currentPiece):
        if y >= 0:
            locked_pos[(x, y)] = currentPiece.color

def validSpace(currentPiece, grid):
    positions = piece.getPiecePositions(currentPiece)

    for x, y in positions:
        if x < 0 or x >= 10 or y >= 20:
            return False

        if y < 0:
            continue

        if grid[y][x] != (0, 0, 0):
            return False

    return True

def deleteRow(grid, locked_pos):
    completed = [
        y for y in range(len(grid))
        if all(cell != (0, 0, 0) for cell in grid[y])
    ]

    if not completed:
        return 0

    for y in completed:
        for x in range(10):
            if (x, y) in locked_pos:
                del locked_pos[(x, y)]

    new_locked_pos = {}

    for (x, y), color in locked_pos.items():
        rows_below = sum(1 for completed_y in completed if completed_y > y)

        new_y = y + rows_below
        new_locked_pos[(x, new_y)] = color

    locked_pos.clear()
    locked_pos.update(new_locked_pos)

    return len(completed)

def getPiece():
    shape = random.choice(piece.shapes)

    cells = piece.getShapeCells(shape, 0)

    shape_width = max(x for x, y in cells) + 1

    board_width = 10

    spawn_x = (board_width - shape_width) // 2
    spawn_y = 0

    return piece.piece(spawn_x, spawn_y, shape)

class Tetris:
    def __init__(self):
        self.locked_pos = {}
        self.grid = createGrid(self.locked_pos)
        self.currentPiece = getPiece()
        self.nextPiece = getPiece()
        self.score = 0

    def reset(self):
        self.locked_pos = {}
        self.grid = createGrid(self.locked_pos)
        self.currentPiece = getPiece()
        self.nextPiece = getPiece()
        self.score = 0

    def startNoRender(self):
        """Starts a tetris game without rendering with pygame."""
     
    def start(self):
        """Starts a tetris game while rendering with pygame."""
        clock = pygame.time.Clock()
        screen = pygame.display.set_mode((screenWidth, screenHeight))
        running = True
        dt = 0

        fall_timer = 0
        fall_speed = 0.5 

        while running:
            dt = clock.tick(60) / 1000

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        self.currentPiece.x -= 1
                        if not validSpace(self.currentPiece, self.grid):
                            self.currentPiece.x += 1

                    elif event.key == pygame.K_RIGHT:
                        self.currentPiece.x += 1
                        if not validSpace(self.currentPiece, self.grid):
                            self.currentPiece.x -= 1

                    elif event.key == pygame.K_SPACE:
                        while validSpace(self.currentPiece, self.grid):
                            self.currentPiece.y += 1

                        self.currentPiece.y -= 1

                        lost = lockPiece(self.currentPiece, self.locked_pos)

                        self.grid = createGrid(self.locked_pos)

                        rows_deleted = deleteRow(self.grid, self.locked_pos)

                        self.grid = createGrid(self.locked_pos)

                        self.currentPiece = self.nextPiece
                        self.nextPiece = getPiece()

                        if not validSpace(self.currentPiece, self.grid):
                            self.reset()
                            fall_timer = 0

                    elif event.key == pygame.K_UP:
                        self.currentPiece.rotation += 1
                        if not validSpace(self.currentPiece, self.grid):
                            self.currentPiece.rotation -= 1

            keys = pygame.key.get_pressed()

            if keys[pygame.K_DOWN]:
                self.currentPiece.y += 1
                if not validSpace(self.currentPiece, self.grid):
                    self.currentPiece.y -= 1                                                                                 

            fall_timer += dt

            screen.fill("black")
            drawGrid(screen, self.grid, self.currentPiece)

            if fall_timer >= fall_speed:
                self.currentPiece.y += 1

                if not validSpace(self.currentPiece, self.grid):
                    self.currentPiece.y -= 1

                    lockPiece(self.currentPiece, self.locked_pos)

                    self.grid = createGrid(self.locked_pos)
                    rows_deleted = deleteRow(self.grid, self.locked_pos)
                    self.grid = createGrid(self.locked_pos)

                    self.currentPiece = self.nextPiece
                    self.nextPiece = getPiece()

                    if not validSpace(self.currentPiece, self.grid):
                        self.reset()
                        fall_timer = 0

                fall_timer = 0

            pygame.display.flip()