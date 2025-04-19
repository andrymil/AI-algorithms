import random
from typing import Callable
from two_player_games.state import State

class MinimaxAlphaBeta:
    """
    Class implementing the minimax algorithm with alpha-beta pruning.

    Args:
        evaluate_fn (Callable[[State], int]): Function to evaluate the state of the game.
    """
    def __init__(self, evaluate_fn: Callable[[State], int]):
        """
        Initializes the MinimaxAlphaBeta class.

        Args:
            evaluate_fn (Callable[[State], int]): Function to evaluate the state of the game.
        """
        self.evaluate_fn = evaluate_fn

    def minimax(self, state: State, depth: int, alpha: int, beta: int, maximizing_player: bool):
        """
        Minimax algorithm with alpha-beta pruning.

        Args:
            state (State): Current state of the game.
            depth (int): Depth of the search tree.
            alpha (int): Alpha value for alpha-beta pruning.
            beta (int): Beta value for alpha-beta pruning.
            maximizing_player (bool): Whether the current player is maximizing.

        Returns:
            Tuple[int, Optional[str]]: Value of the best move and the best move.
        """
        if depth == 0 or state.is_finished():
            return self.evaluate_fn(state, maximizing_player), None

        best_moves = []
        if maximizing_player:
            best_value = float('-inf')
            for move in state.get_moves():
                new_state = state.make_move(move)
                value, _ = self.minimax(new_state, depth - 1, alpha, beta, False)
                if value > best_value:
                    best_value = value
                    best_moves = [move]
                elif value == best_value:
                    best_moves.append(move)
                alpha = max(alpha, best_value)
                if beta <= alpha:
                    break
            return best_value, random.choice(best_moves)
        else:
            best_value = float('inf')
            for move in state.get_moves():
                new_state = state.make_move(move)
                value, _ = self.minimax(new_state, depth - 1, alpha, beta, True)
                if value < best_value:
                    best_value = value
                    best_moves = [move]
                elif value == best_value:
                    best_moves.append(move)
                beta = min(beta, best_value)
                if beta <= alpha:
                    break
            return best_value, random.choice(best_moves)
