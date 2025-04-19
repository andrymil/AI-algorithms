import time
import numpy as np
import matplotlib.pyplot as plt
from play import PlayNim
from two_player_games.state import State


def evaluate_fn(state: State, maximizing_player: bool) -> int:
    """
    Evaluation function for the Nim game.

    Args:
        state (State): Current state of the game.
        maximizing_player (bool): Whether the current player is maximizing.

    Returns:
        int: Evaluation value of the state.
    """
    if state.is_finished():
        return 1000 if maximizing_player else -1000

    nim_sum = 0
    for heap in state.heaps:
        nim_sum ^= heap
    heap_count = sum(1 for heap in state.heaps if heap > 0)
    return (1 if nim_sum != 0 else -1) + (0.1 * heap_count)


def run_experiments(initial_heaps, evaluate_fn, n_games=100):
    """
    Function to run experiments for different search depths.
    """
    depths = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15]
    n_games = 100
    results_matrix_player1 = np.zeros((len(depths), len(depths)))
    results_matrix_player2 = np.zeros((len(depths), len(depths)))

    for i, player1_depth in enumerate(depths):
        for j, player2_depth in enumerate(depths):
            nim_game = PlayNim(initial_heaps, evaluate_fn, player1_depth, player2_depth)
            results = nim_game.play_n_games(n_games)
            results_matrix_player1[i, j] = results["Player 1"] / n_games
            results_matrix_player2[i, j] = results["Player 2"] / n_games

    fig, ax = plt.subplots()
    cax = ax.matshow(results_matrix_player1, cmap="viridis")
    fig.colorbar(cax)

    ax.set_xticks(range(len(depths)))
    ax.set_yticks(range(len(depths)))
    ax.set_xticklabels(depths)
    ax.set_yticklabels(depths)
    plt.xlabel("Depth of Player 2")
    plt.ylabel("Depth of Player 1")
    plt.title("Win percentage of Player 1 depending on search depth")
    plt.savefig("experiment_results_player1_depths_1_to_15___2.png")
    plt.show()

    fig, ax = plt.subplots()
    cax = ax.matshow(results_matrix_player2, cmap="viridis")
    fig.colorbar(cax)

    ax.set_xticks(range(len(depths)))
    ax.set_yticks(range(len(depths)))
    ax.set_xticklabels(depths)
    ax.set_yticklabels(depths)
    plt.xlabel("Depth of Player 2")
    plt.ylabel("Depth of Player 1")
    plt.title("Win percentage of Player 2 depending on search depth")
    plt.savefig("experiment_results_player2_depths_1_to_15___2.png")
    plt.show()


def analyze_execution_time(initial_heaps, evaluate_fn, max_depth):
    """
    Function to analyze the execution time of the minimax algorithm for different depths.

    Args:
        initial_heaps (Tuple[int, int, int]): Initial heaps for the Nim game.
        evaluate_fn (Callable[[State, bool], int]): Evaluation function for the Nim game.
        max_depth (int): Maximum depth to analyze.
    """
    times = []
    for depth in range(1, max_depth + 1):
        nim_game = PlayNim(
            initial_heaps, evaluate_fn, player1_depth=depth, player2_depth=depth
        )

        start_time = time.time()
        nim_game.play_single_game()
        end_time = time.time()

        execution_time = end_time - start_time
        times.append(execution_time)
        print(f"Depth: {depth}, Execution Time: {execution_time:.4f} seconds")

    plt.plot(range(1, max_depth + 1), times, marker="o")
    plt.xlabel("Search Depth")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Execution Time of Minimax with Alpha-Beta Pruning")
    plt.grid(True)
    plt.savefig("execution_time_analysis___3.png")
    plt.show()


def play_against_random(initial_heaps, evaluate_fn, max_depth, n_games=100):
    """
    Function to play against a random player and analyze the win percentage.

    Args:
        initial_heaps (Tuple[int, int, int]): Initial heaps for the Nim game.
        evaluate_fn (Callable[[State, bool], int]): Evaluation function for the Nim game.
        max_depth (int): Maximum depth for the Minimax player.
        n_games (int): Number of games to play.
    """
    depths = list(range(1, max_depth + 1))
    win_ratios = []

    for depth in depths:
        nim_game = PlayNim(
            initial_heaps, evaluate_fn, player1_depth=depth, player2_depth=0
        )  # Depth 0 for a random player
        results = nim_game.play_n_games(n_games)
        win_ratios.append(results["Player 1"] / n_games)
        print(
            f"Depth: {depth}, Player 1 Win Percentage: {results['Player 1'] / n_games:.2f}"
        )

    plt.plot(depths, win_ratios, marker="o", color="r")
    plt.xlabel("Depth of Minimax Player")
    plt.ylabel("Win Percentage")
    plt.title("Win Percentage of Minimax Player Against Random Opponent")
    plt.grid(True)
    plt.savefig("win_against_random_analysis____3.png")
    plt.show()


initial_heaps = (3, 4, 5)

run_experiments(initial_heaps, evaluate_fn)

play_against_random(initial_heaps, evaluate_fn, max_depth=10)

analyze_execution_time(initial_heaps, evaluate_fn, max_depth=10)
