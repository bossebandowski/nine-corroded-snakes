import argparse
import numpy as np
import sys


def load_level(level_path: str) -> np.ndarray:
    """Load a .csv file at arg and parse to numpy array.

    Args:
        level_path (str): path to a csv file with valid format.

    Returns:
        np.ndarray: the level in its numpy format
    """
    return np.genfromtxt(level_path, delimiter=",", dtype=np.int8)


def convert_to_base_0(level) -> None:
    clues = np.argwhere(level > 0)
    for x, y in clues:
        level[x, y] = level[x, y] - 1


def convert_to_base_1(level) -> None:
    clues = np.argwhere(level >= 0)
    for x, y in clues:
        level[x, y] = level[x, y] + 1


def parse_args():
    parser = argparse.ArgumentParser(description="Solve sudokus.")
    parser.add_argument(
        "--level",
        help="Path to level to be solved",
        required=True,
    )
    return parser.parse_args(sys.argv[1:])


def generate_options() -> np.ndarray:
    return np.ones((9, 9, 9), dtype=bool)


def remove_init_hints(level: np.ndarray, options: np.ndarray) -> None:
    clues = np.argwhere(level >= 0)
    for x, y in clues:
        options[x, y] = np.zeros((1, 1, 9))
        options[x, y, level[x, y]] = 1


def remove_squares(level: np.ndarray, options: np.ndarray) -> None:
    for square_id in range(9):
        orin_x = int(square_id / 3) * 3
        orin_y = (square_id % 3) * 3
        square = level[orin_x : orin_x + 3, orin_y : orin_y + 3]
        for x in range(3):
            for y in range(3):
                for existing_value in square[square >= 0]:
                    options[orin_x + x, orin_y + y, existing_value] = 0


def remove_rows(level: np.ndarray, options: np.ndarray) -> None:
    for row_id in range(9):
        row = level[row_id, :]
        for col in range(9):
            for existing_value in row[row >= 0]:
                options[row_id, col, existing_value] = 0


def remove_columns(level: np.ndarray, options: np.ndarray) -> None:
    for col_id in range(9):
        col = level[:, col_id]

        for row in range(9):
            for existing_value in col[col >= 0]:
                options[row, col_id, existing_value] = 0


def fill_options(level: np.ndarray, options: np.ndarray) -> None:
    for row in range(9):
        for col in range(9):
            if np.sum(options[row, col]) == 1:
                level[row, col] = np.argmax(options[row, col])


def solve(level: np.ndarray) -> np.ndarray:
    options = generate_options()
    unique, counts = np.unique(level, return_counts=True)
    prev_unknowns = dict(zip(unique, counts))[-1]
    cur_unknowns = prev_unknowns - 1

    while cur_unknowns < prev_unknowns:
        prev_unknowns = cur_unknowns
        remove_init_hints(level, options)
        remove_squares(level, options)
        remove_rows(level, options)
        remove_columns(level, options)

        fill_options(level, options)

        unique, counts = np.unique(level, return_counts=True)
        if -1 in dict(zip(unique, counts)):
            cur_unknowns = dict(zip(unique, counts))[-1]
        else:
            cur_unknowns = 0

    return level


def main() -> None:
    parsed_args = parse_args()
    level = load_level(parsed_args.level)
    convert_to_base_0(level)
    solved_level = solve(level)
    convert_to_base_1(solved_level)
    print(solved_level)


if __name__ == "__main__":
    main()
