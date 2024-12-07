import random
from minimax import MinimaxAlphaBeta
from two_player_games.games.nim import Nim


class PlayNim:
    """
    Class to play Nim games between two players with different search depths.

    Args:
        initial_heaps (Tuple[int, int, int]): Initial heaps for the Nim game.
        evaluate_fn (Callable[[State, bool], int]): Evaluation function for the Nim game.
        player1_depth (int): Search depth for Player 1.
        player2_depth (int): Search depth for Player 2.
    """

    def __init__(self, initial_heaps, evaluate_fn, player1_depth, player2_depth):
        """
        Initializes the PlayNim class.

        Args:
            initial_heaps (Tuple[int, int, int]): Initial heaps for the Nim game.
            evaluate_fn (Callable[[State, bool], int]): Evaluation function for the Nim game.
            player1_depth (int): Search depth for Player 1.
            player2_depth (int): Search depth for Player 2.
        """
        self.initial_heaps = initial_heaps
        self.evaluate_fn = evaluate_fn
        self.player1_depth = player1_depth
        self.player2_depth = player2_depth
        self.game = Nim(initial_heaps)
        self.minimax_algorithm = MinimaxAlphaBeta(evaluate_fn)

    def reset_game(self):
        """
        Resets the game to the initial state.
        """
        self.game = Nim(self.initial_heaps)

    def play_single_game(self):
        """
        Plays a single game of Nim between two players with different search depths.

        Returns:
            Player: Winner of the game.
        """
        while not self.game.is_finished():
            current_player = self.game.get_current_player()
            if current_player.char == "1":

                _, move = self.minimax_algorithm.minimax(
                    state=self.game.state,
                    depth=self.player1_depth,
                    alpha=float("-inf"),
                    beta=float("inf"),
                    maximizing_player=True,
                )
            else:
                if self.player2_depth == 0:
                    # Second player does a random move
                    possible_moves = list(self.game.state.get_moves())
                    if possible_moves:
                        move = random.choice(possible_moves)
                    else:
                        move = None
                else:
                    _, move = self.minimax_algorithm.minimax(
                        state=self.game.state,
                        depth=self.player2_depth,
                        alpha=float("-inf"),
                        beta=float("inf"),
                        maximizing_player=True,
                    )
            if move is not None:
                self.game.make_move(move)
            else:
                raise ValueError("No valid move found for the current state.")

        return self.game.get_winner()

    def play_n_games(self, n):
        """
        Plays n games of Nim between two players with different search depths.

        Args:
            n (int): Number of games to play.

        Returns:
            Dict[str, int]: Results of the games.
        """
        results = {"Player 1": 0, "Player 2": 0}
        for _ in range(n):
            winner = self.play_single_game()
            if winner.char == "1":
                results["Player 1"] += 1
            else:
                results["Player 2"] += 1
            self.reset_game()
        return results
