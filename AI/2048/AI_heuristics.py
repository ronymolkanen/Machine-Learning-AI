import constants as c
import logic

commands = {
    c.KEY_UP: logic.up,
    c.KEY_DOWN: logic.down,
    c.KEY_LEFT: logic.left,
    c.KEY_RIGHT: logic.right
}

def snake_heuristic(matrix):
    # INF = 2**64
    PERFECT_SNAKE = [
        [2, 2**2, 2**3, 2**4],
        [2**8, 2**7, 2**6, 2**5],
        [2**9, 2**10, 2**11, 2**12],
        [2**16, 2**15, 2**14, 2**13]]

    snake_score = 0
    for i in range(c.GRID_LEN):
        for j in range(c.GRID_LEN):
            # Lasketaan kuinka paljon arvo poikkeaa täydellisestä snake-muodosta
            snake_score += matrix[i][j] * PERFECT_SNAKE[i][j]
    return snake_score


def smoothness(matrix):
    smoothness_score = 0
    for i in range(c.GRID_LEN):
        for j in range(c.GRID_LEN):
            if matrix[i][j] != 0:
                value = matrix[i][j]
                if i < c.GRID_LEN - 1 and matrix[i + 1][j] != 0:
                    smoothness_score -= abs(value - matrix[i + 1][j])
                if j < c.GRID_LEN - 1 and matrix[i][j + 1] != 0:
                    smoothness_score -= abs(value - matrix[i][j + 1])
    return smoothness_score

def free_cells(matrix):
    return sum(row.count(0) for row in matrix)

def max_tile_in_corner(matrix):
    max_tile = max(max(row) for row in matrix)
    corners = [matrix[0][0], matrix[0][c.GRID_LEN - 1], matrix[c.GRID_LEN - 1][0], matrix[c.GRID_LEN - 1][c.GRID_LEN - 1]]
    return max_tile if max_tile in corners else 0

def monotonicity(matrix):
    score = 0
    for row in matrix:
        score += sum(row[i] <= row[i + 1] for i in range(len(row) - 1))
        score += sum(row[i] >= row[i + 1] for i in range(len(row) - 1))
    return score

def merge_potential(matrix):
    merge_score = 0
    for i in range(c.GRID_LEN):
        for j in range(c.GRID_LEN):
            if matrix[i][j] != 0:
                if i < c.GRID_LEN - 1 and matrix[i][j] == matrix[i + 1][j]:
                    merge_score += 1
                if j < c.GRID_LEN - 1 and matrix[i][j] == matrix[i][j + 1]:
                    merge_score += 1
    return merge_score

def heuristic(matrix):
    return (0.1 * smoothness(matrix) +
            10.0 * free_cells(matrix) +
            100.0 * max_tile_in_corner(matrix) +
            0.5 * monotonicity(matrix) +
            1.0 * merge_potential(matrix) +
            0.5 * snake_heuristic(matrix)) 

