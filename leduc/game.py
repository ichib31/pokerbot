from itertools import permutations
from typing import NamedTuple, Optional, Tuple

from core.game import Game

# Cards are ints 0..5: two suits x three ranks. rank = card % 3 (0=J, 1=Q, 2=K).
DECK = list(range(6))


def _rank(card: int) -> int:
    return card % 3


class LeducState(NamedTuple):
    p0_card: Optional[int]
    p1_card: Optional[int]
    board: Optional[int]
    round: int  # 1 or 2
    history: str  # actions this round: 'k' check, 'b' bet, 'c' call, 'r' raise, 'f' fold
    contributed: Tuple[int, int]  # total chips each player has put in (incl. ante)


class LeducGame(Game):
    def initial_state(self) -> LeducState:
        return LeducState(p0_card=None, p1_card=None, board=None, round=1, history="", contributed=(0, 0))

    def is_terminal(self, state: LeducState) -> bool:
        if state.p0_card is None:
            return False
        h = state.history
        if h and h[-1] == "f":
            return True
        return state.round == 2 and bool(h) and (h[-1] == "c" or h == "kk")

    def is_chance(self, state: LeducState) -> bool:
        if state.p0_card is None:
            return True
        h = state.history
        return state.round == 1 and bool(h) and (h[-1] == "c" or h == "kk")

    def current_player(self, state: LeducState) -> int:
        return len(state.history) % 2

    def actions(self, state: LeducState) -> list:
        h = state.history
        if h == "" or h[-1] == "k":
            return ["k", "b"]
        acts = ["f", "c"]
        if h.count("r") < 2:
            acts.append("r")
        return acts

    def apply(self, state: LeducState, action: str) -> LeducState:
        player = self.current_player(state)
        unit = 2 if state.round == 1 else 4
        c0, c1 = state.contributed
        if action == "b":
            c0, c1 = (c0 + unit, c1) if player == 0 else (c0, c1 + unit)
        elif action == "c":
            target = max(c0, c1)
            c0, c1 = (target, c1) if player == 0 else (c0, target)
        elif action == "r":
            target = max(c0, c1) + unit
            c0, c1 = (target, c1) if player == 0 else (c0, target)
        return state._replace(history=state.history + action, contributed=(c0, c1))

    def chance_outcomes(self, state: LeducState) -> list:
        if state.p0_card is None:
            deals = list(permutations(DECK, 2))
            prob = 1 / len(deals)
            return [
                (LeducState(p0_card=a, p1_card=b, board=None, round=1, history="", contributed=(1, 1)), prob)
                for a, b in deals
            ]
        remaining = [c for c in DECK if c not in (state.p0_card, state.p1_card)]
        prob = 1 / len(remaining)
        return [(state._replace(board=c, round=2, history=""), prob) for c in remaining]

    def payoff(self, state: LeducState) -> float:
        h = state.history
        c0, c1 = state.contributed
        if h and h[-1] == "f":
            folder = (len(h) - 1) % 2
            return -c0 if folder == 0 else c1

        p0_rank, p1_rank, board_rank = _rank(state.p0_card), _rank(state.p1_card), _rank(state.board)
        p0_pairs = p0_rank == board_rank
        p1_pairs = p1_rank == board_rank
        if p0_pairs and not p1_pairs:
            winner = 0
        elif p1_pairs and not p0_pairs:
            winner = 1
        elif p0_rank > p1_rank:
            winner = 0
        elif p1_rank > p0_rank:
            winner = 1
        else:
            winner = None

        if winner == 0:
            return c1
        if winner == 1:
            return -c0
        return 0

    def infoset_key(self, state: LeducState) -> str:
        player = self.current_player(state)
        own = state.p0_card if player == 0 else state.p1_card
        board = state.board if state.board is not None else -1
        return f"{own}|{board}|r{state.round}|{state.history}"

    def num_actions(self, state: LeducState) -> int:
        return len(self.actions(state))
